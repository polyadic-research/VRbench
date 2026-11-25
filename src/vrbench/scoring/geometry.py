"""FCL solid collision primitives; meshes exported as per-geom convex hulls."""
import numpy as np
import fcl
from scipy.spatial import ConvexHull


def shape(spec):
    kind = spec["type"]
    s = spec.get("size", [])
    if kind == "box": return fcl.Box(*[2*x for x in s[:3]])
    if kind == "sphere": return fcl.Sphere(s[0])
    if kind == "capsule": return fcl.Capsule(s[0], 2*s[1])
    if kind == "cylinder": return fcl.Cylinder(s[0], 2*s[1])
    if kind == "ellipsoid": return fcl.Ellipsoid(*s[:3])
    if kind == "plane": return fcl.Plane(0, 0, 1, 0)
    if kind == "convex":
        vertices = np.asarray(spec["vertices"], dtype=float)
        hull = ConvexHull(vertices)
        faces = []
        for tri, equation in zip(hull.simplices, hull.equations):
            tri = tri.copy()
            p = vertices[tri]
            if np.dot(np.cross(p[1]-p[0], p[2]-p[0]), equation[:3]) < 0:
                tri = tri[::-1]
            faces.extend([3, *map(int, tri)])
        return fcl.Convex(vertices, len(hull.simplices), np.asarray(faces, dtype=np.int32))
    raise ValueError(f"Unsupported collision shape {kind}; no silent box fallback")


class Geometry:
    def __init__(self, specs):
        self.specs = {s["name"]: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError("Geometry names must be unique")
        self.objects = {name: fcl.CollisionObject(shape(s)) for name, s in self.specs.items()}
        self.world = {}
        self.bounds = {}
        self.local_bounds = {name: self._local_bounds(spec) for name, spec in self.specs.items()}
        self._contact_requests = {
            1: fcl.CollisionRequest(num_max_contacts=1, enable_contact=True),
            16: fcl.CollisionRequest(num_max_contacts=16, enable_contact=True),
        }
        self._boolean_request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)

    @staticmethod
    def _local_bounds(spec):
        kind = spec["type"]
        size = np.asarray(spec.get("size", []), dtype=float)
        if kind == "plane":
            return None
        if kind == "convex":
            vertices = np.asarray(spec["vertices"], dtype=float)
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            return (low + high) / 2, (high - low) / 2
        if kind == "sphere": half = np.repeat(size[0], 3)
        elif kind == "capsule": half = np.array([size[0], size[0], size[0] + size[1]])
        elif kind == "cylinder": half = np.array([size[0], size[0], size[1]])
        else: half = size[:3]
        return np.zeros(3), half

    def update(self, transforms):
        for name, t in transforms.items():
            if self.specs[name].get("group")=="static" and name in self.world:
                continue
            self.world[name]=t.copy()
            self.objects[name].setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
            local = self.local_bounds[name]
            if local is None:
                self.bounds[name] = (np.full(3, -np.inf), np.full(3, np.inf))
                continue
            center, half = local
            c = t[:3, :3] @ center + t[:3, 3]
            h = np.abs(t[:3, :3]) @ half
            self.bounds[name] = (c-h, c+h)

    def broad_pairs(self, left, right, self_pairs=False):
        if not right: return
        lo = np.array([self.bounds[n][0] for n in right])
        hi = np.array([self.bounds[n][1] for n in right])
        for i, a in enumerate(left):
            amin, amax = self.bounds[a]
            for k in np.flatnonzero(np.all(lo <= amax, axis=1) & np.all(hi >= amin, axis=1)):
                b = right[k]
                if a != b and (not self_pairs or i < k):
                    yield a, b

    def distance(self, a, b):
        return float(fcl.distance(self.objects[a], self.objects[b], fcl.DistanceRequest(), fcl.DistanceResult()))

    def minimum_projection(self,name,normal):
        s=self.specs[name];t=self.world[name];n=t[:3,:3].T@np.asarray(normal);z=np.asarray(s.get("size",[]));kind=s["type"]
        if kind=="convex":offset=float(np.min(np.asarray(s["vertices"])@n))
        elif kind=="box":offset=-float(abs(n)@z[:3])
        elif kind=="sphere":offset=-float(z[0])
        elif kind=="capsule":offset=-float(z[0]+z[1]*abs(n[2]))
        elif kind=="cylinder":offset=-float(z[0]*np.linalg.norm(n[:2])+z[1]*abs(n[2]))
        elif kind=="ellipsoid":offset=-float(np.linalg.norm(z[:3]*n))
        else:raise ValueError("Target support projection requires a finite solid shape")
        return float(t[:3,3]@normal)+offset

    def collide_any(self, a, b):
        """Fast boolean query; it intentionally does not claim a depth."""
        return bool(fcl.collide(self.objects[a], self.objects[b], self._boolean_request,
                                fcl.CollisionResult()))

    def collide(self, a, b, max_contacts=16):
        result = fcl.CollisionResult()
        request = self._contact_requests.get(max_contacts)
        if request is None:
            request = fcl.CollisionRequest(num_max_contacts=max_contacts, enable_contact=True)
            self._contact_requests[max_contacts] = request
        n = fcl.collide(self.objects[a], self.objects[b], request, result)
        depth = max((c.penetration_depth for c in result.contacts), default=0.0)
        return bool(n), float(depth)
