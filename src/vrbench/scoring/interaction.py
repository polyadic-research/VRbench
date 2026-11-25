"""Pose-conditioned contact evidence from the supplied gripper and task geometry."""
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from . import manipulation as motion


def convex(vertices):
    import fcl
    points = np.unique(np.asarray(vertices, float), axis=0)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    v = points[faces]
    inward = np.einsum('ij,ij->i', np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]), hull.equations[:,:3]) < 0
    faces[inward] = faces[inward][:, ::-1]
    encoded = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return fcl.Convex(points, len(faces), encoded)


def vertices(region):
    if region['type'] == 'convex_polytope':
        return np.array(region['vertices_world'])
    if region['type'] == 'box':
        pose = np.array(region['world_pose'])
        corners = np.array(list(product((-1, 1), repeat=3))) * region['size']
        return corners @ pose[:3,:3].T + pose[:3,3]
    if region['type'] == 'cylinder':
        angles = np.arange(64) * (2*np.pi/64)
        xy = np.stack((np.cos(angles),np.sin(angles)),axis=1)*region['radius']
        local = np.concatenate([np.column_stack((xy,np.full(len(xy),z))) for z in (-region['half_height'],region['half_height'])])
        pose=np.array(region['world_pose'])
        return local @ pose[:3,:3].T + pose[:3,3]
    raise ValueError('Unsupported interaction region: '+region['type'])


def physical(regions):
    core = [r for r in regions if 'core' in r.get('name','')]
    return core or regions


class Trace:
    def __init__(self, prediction, initial, config):
        import fcl
        self.fcl = fcl
        self.config = config
        self.ps,self.poses,events = motion.trajectory(prediction,initial)
        self.cycles = motion.grip_cycles(events)
        self.closed = np.zeros(len(events),dtype=bool)
        for c,e in self.cycles:
            self.closed[c:e]=True
        self.rotations=Rotation.from_matrix(self.poses[:,:3,:3])
        self.slerp=Slerp(np.arange(len(events)),self.rotations)
        self.jaws=[]
        self.parts=[]
        for part in config['gripper']:
            xyz=np.array(part['closed_vertices'])
            shift=np.array(part['open_translation'])
            if part['pad']:
                self.jaws.append(convex(np.vstack((xyz,xyz+shift))))
            self.parts.append((convex(xyz),shift,part['name']))
        self.cache={}
        self.travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.ps,axis=0),axis=1))]
        self.slide=None
        self.centering={}
        self.jaw_axis=np.array(next(p for p in config["gripper"] if p["pad"])["open_translation"])/config["jaw_travel_m"]

    def solids(self,regions):
        key=id(regions)
        if key not in self.cache:
            objs=[]
            for r in physical(regions):
                v=vertices(r);origin=v.mean(axis=0)
                objs.append(self.fcl.CollisionObject(convex(v-origin),self.fcl.Transform(origin)))
            self.cache[key]=(regions,objs)
        return self.cache[key][1]

    def gap(self,shape,R,p,objects):
        f=self.fcl
        moving=f.CollisionObject(shape,f.Transform(np.ascontiguousarray(R),np.ascontiguousarray(p)))
        return max(0.,min(f.distance(moving,target) for target in objects))

    def grip(self,regions,close,release,shift=None,fixed=False):
        """Both opposing pad sweeps must reach the same target during closure."""
        end=len(self.ps) if release is None else release
        end=min(end,np.searchsorted(self.travel,self.travel[close]+self.config['jaw_travel_m'],side='right'))
        target=self.solids(regions)
        best=(float('inf'),close)
        for k in range(close,max(close+1,end)):
            R=self.poses[k,:3,:3];p=self.ps[k] if shift is None else self.ps[k]-shift
            ds,offset=self.capture(R,p,target,fixed)
            self.centering[id(regions),k]=offset
            if ds<best[0]:best=(ds,k)
            if ds<=1e-9:break
        return best

    def capture(self,R,p,target,fixed=False):
        gaps=[self.gap(shape,R,p,target) for shape in self.jaws]
        if fixed or min(gaps)>1e-9 or max(gaps)<=1e-9:
            return max(gaps),np.zeros(3)
        axis=R@self.jaw_axis*(1 if gaps[0]>gaps[1] else -1)
        stroke=self.config['jaw_travel_m']
        for travel in np.linspace(0,stroke,17)[1:]:
            if max(self.gap(q,R,p-axis*travel,target) for q in self.jaws)<=1e-9:
                lo,hi=travel-stroke/16,travel
                for _ in range(12):
                    mid=(lo+hi)/2
                    if max(self.gap(q,R,p-axis*mid,target) for q in self.jaws)<=1e-9:hi=mid
                    else:lo=mid
                return 0.,axis*hi
        return max(gaps),np.zeros(3)

    def clock(self,step=None):
        if not step:return np.arange(len(self.ps),dtype=float)
        out=[0.]
        for i,delta in enumerate(np.diff(self.ps,axis=0)):
            n=max(1,int(np.ceil(np.linalg.norm(delta)/step)))
            out.extend(i+np.arange(1,n+1)/n)
        return np.array(out)

    def drawer_pull(self,regions):
        axis=np.asarray(self.config['drawer_outward_axis'])
        best=(1e6,None,None,None)
        for c,e in self.cycles:
            gap,k=self.grip(regions,c,e,fixed=True)
            stop=len(self.ps) if e is None else e+1
            travel=(self.ps[k:stop]-self.ps[k])@axis
            j=k+int(np.argmax(travel))
            distance = travel[-1] if self.config.get('drawer_direction_policy') == 'held_net' else travel[j-k]
            gap=max(gap,.003-float(distance))
            if gap<best[0]:best=(gap,k,e,j)
            if gap<=1e-9:break
        return best

    def drawer_shift(self,close):
        cfg=self.config.get('drawer')
        if cfg is None:return None
        if self.slide is None:
            axis=np.array(cfg['axis']);targets=self.solids(cfg['handle']);offset=0.;offsets=[0.]
            for i in range(1,len(self.ps)):
                p=self.ps[i-1]-axis*offset;R=self.poses[i-1,:3,:3]
                if any(self.gap(shape,R,p+R@(np.zeros(3) if self.closed[i-1] else shift),targets)<=1e-9 for shape,shift,name in self.parts):
                    offset=np.clip(offset+np.dot(self.ps[i]-self.ps[i-1],axis),*cfg['limits'])
                offsets.append(offset)
            self.slide=np.outer(offsets,axis)
        return self.slide[close]

    def series(self,regions,clock,action,scale=.05):
        ds=np.full(len(clock),1e6)
        if action['kind']=='grasp':
            for c,e in self.cycles:
                shift=self.drawer_shift(c) if action.get('in_drawer') else None
                gap,k=self.grip(regions,c,e,shift)
                j=np.searchsorted(clock,k)
                if j<len(ds) and clock[j]==k:ds[j]=min(ds[j],gap)
            return ds
        objects=self.solids(regions)
        rots=self.slerp(clock).as_matrix()
        ps=np.column_stack([np.interp(clock,np.arange(len(self.ps)),col) for col in self.ps.T])
        cfg=action['joint']
        axis=np.array(cfg['axis']);pivot=np.array(cfg['pivot'])
        if action.get('direction_policy') == 'contact_window':
            if action['kind'] == 'hinge_path':
                return self.finger_contact_series(objects, clock, ps, rots, action)[0].min(axis=0)
            window = self.contact_window_progress(objects, clock, ps, rots, action, scale)
            default_action = dict(action)
            del default_action['direction_policy']
            return np.minimum(window, self.series(regions, clock, default_action, scale))
        if action.get('contact_turn_trend'):
            return self.turn_trend(objects, clock, ps, rots, cfg)
        vertices_all=np.vstack([vertices(r) for r in physical(regions)])
        center=vertices_all.mean(axis=0)
        for i,(p,R) in enumerate(zip(ps,rots)):
            if np.linalg.norm(p-center)>np.max(np.linalg.norm(vertices_all-center,axis=1))+.16:continue
            opened=not self.closed[int(clock[i])]
            if action.get('bilateral_grip'):
                if opened:
                    continue
                grip_gap, _ = self.capture(R, p, objects, fixed=True)
            for shape,shift,name in self.parts:
                local=np.array(self.config['part_centers'][name])+shift*opened
                pt=p+R@local
                q=p+R@(shift*opened)
                gap=max(0., self.gap(shape,R,q,objects)-action.get('contact_tolerance_m', 0.))
                reach=self.config['jaw_travel_m']
                lo=np.searchsorted(self.travel,np.interp(clock[i],np.arange(len(self.ps)),self.travel)-reach)
                hi=np.searchsorted(self.travel,np.interp(clock[i],np.arange(len(self.ps)),self.travel)+reach,side='right')
                lo=min(lo,int(np.floor(clock[i])));hi=max(hi,int(np.ceil(clock[i]))+1)
                direction=axis if cfg['type']=='slide' else np.cross(axis,pt-pivot)
                mag=np.linalg.norm(direction)
                if mag<1e-9:continue
                direction=direction/mag*cfg['sign']
                nearby=self.ps[lo:hi]+np.einsum('nij,j->ni',self.poses[lo:hi,:3,:3],local)
                a=nearby[:int(clock[i])-lo+1];b=nearby[int(np.ceil(clock[i]))-lo:]
                progress=max(0.,float(np.max((pt-a)@direction)),float(np.max((b-pt)@direction))) if len(a) and len(b) else 0.
                if action.get('bilateral_grip'):
                    end = next((j for j in range(int(np.ceil(clock[i])),hi) if not self.closed[j]), hi)
                    following = b[:max(0, end-int(np.ceil(clock[i])))]
                    progress = max(0., float(np.max((following-pt)@direction))) if len(following) else 0.
                    gap = max(gap, grip_gap)
                needed=cfg['min_motion_m']
                if progress>1e-9:
                    ds[i]=min(ds[i],gap-scale*np.log(min(1.,progress/needed)))
        return ds


    def finger_contact_series(self, objects, clock, positions, rotations, action):
        """Finger gaps and centers in the predicted aperture at each sample."""
        opened = ~self.closed[clock.astype(int)]
        tolerance = action.get('contact_tolerance_m', 0.)
        gaps, centers = [], []
        for shape, shift, name in self.parts:
            offsets = opened[:,None]*shift
            local = offsets + self.config['part_centers'][name]
            centers.append(positions + np.einsum('nij,nj->ni', rotations, local))
            origins = positions + np.einsum('nij,nj->ni', rotations, offsets)
            gaps.append([max(0., self.gap(shape, R, p, objects)-tolerance)
                         for R,p in zip(rotations,origins)])
        gaps = np.asarray(gaps)
        if action.get('bilateral_grip'):
            grip = [1e6 if opened[i] else self.capture(R, p, objects, fixed=True)[0]
                    for i,(R,p) in enumerate(zip(rotations,positions))]
            gaps = np.maximum(gaps, grip)
        return gaps, np.asarray(centers)

    def contact_window_progress(self, objects, clock, positions, rotations, action, scale):
        joint = action['joint']
        axis = np.asarray(joint['axis'])
        gap, centers = self.finger_contact_series(objects, clock, positions, rotations, action)
        if len(clock) < 2:
            return np.full(len(clock), 1e6)
        if joint['type'] == 'slide':
            motion = np.einsum('fvi,i->fv', np.diff(centers,axis=1), axis)
        else:
            radial = centers-np.asarray(joint['pivot'])
            radial -= np.einsum('fvi,i->fv', radial, axis)[:,:,None]*axis
            left, right = radial[:,:-1], radial[:,1:]
            angle = np.arctan2(np.einsum('fvi,i->fv',np.cross(left,right),axis),
                               np.einsum('fvi,fvi->fv',left,right))
            motion = angle*np.minimum(np.linalg.norm(left,axis=2),np.linalg.norm(right,axis=2))
        motion *= joint['sign']
        pair_gap = np.maximum(gap[:,:-1],gap[:,1:])
        chosen = np.argmin(pair_gap,axis=0)
        ix = np.arange(len(chosen))
        increments, gaps = motion[chosen,ix], pair_gap[chosen,ix]
        closed = self.closed[clock.astype(int)]
        errors = np.full(len(clock),1e6)
        minimum = joint['min_motion_m']
        for band in (1e-9,.002,.005,.010,.020,.050):
            total = floor = 0.
            for k,(delta,distance) in enumerate(zip(increments,gaps),1):
                if distance > band or closed[k] != closed[k-1]:
                    total = floor = 0.
                    continue
                total += delta
                floor = min(floor,total)
                progress = total-floor
                if progress > 1e-9:
                    penalty = 0. if band <= 1e-9 else band
                    errors[k] = min(errors[k], penalty-scale*np.log(min(1.,progress/minimum)))
        return errors

    def turn_trend(self, objects, clock, positions, rotations, joint):
        """Signed tangential travel of one finger while it is at the spout."""
        errors = np.full(len(clock), 1e6)
        axis, pivot = np.asarray(joint['axis']), np.asarray(joint['pivot'])
        for shape, shift, name in self.parts:
            opened = ~self.closed[clock.astype(int)]
            offsets = opened[:,None]*shift
            centers = positions + np.einsum('nij,nj->ni', rotations, offsets+self.config['part_centers'][name])
            translations = positions + np.einsum('nij,nj->ni', rotations, offsets)
            gaps = np.array([self.gap(shape,R,p,objects) for R,p in zip(rotations,translations)])
            radial = centers-pivot
            radial -= np.outer(np.einsum('ij,j->i',radial,axis),axis)
            angles = np.arctan2(np.einsum('ij,j->i',np.cross(radial[:-1],radial[1:]),axis),
                                np.einsum('ij,ij->i',radial[:-1],radial[1:]))
            arcs = joint['sign']*angles*np.minimum(np.linalg.norm(radial[:-1],axis=1),
                                                    np.linalg.norm(radial[1:],axis=1))
            left = 0
            total = 0.
            for end, arc in enumerate(arcs,1):
                detached = joint.get('require_continuous_contact') and max(gaps[end-1],gaps[end]) > 1e-9
                if detached or arc < -1e-10 or opened[end] != opened[end-1]:
                    left, total = end, 0.
                    continue
                total += max(0.,arc)
                if total < joint['min_motion_m']:
                    continue
                while left < end and total-max(0.,arcs[left]) >= joint['min_motion_m']:
                    total -= max(0.,arcs[left])
                    left += 1
                errors[end] = min(errors[end], float(max(gaps[left:end+1])))
        return errors

    def hinge_path(self, regions, clock, start, action, scale):
        """Match ordered handle poses after prerequisites, on a common interpolated clock."""
        fine = self.clock(action['step_m'])
        if start < len(clock):
            fine = fine[fine >= clock[start]]
        else:
            fine = fine[:0]
        errors = np.full(len(clock), 1e6)
        if not len(fine):
            return errors, {}
        prev = None
        parents = []
        sizes = action.get('pose_group_sizes', [1]*len(regions))
        if any(size < 1 for size in sizes) or sum(sizes) != len(regions):
            raise ValueError('Invalid hinge contact pose groups')
        offset = 0
        travel = np.interp(fine, np.arange(len(self.ps)), self.travel)
        for group, size in enumerate(sizes):
            contact = action
            if group == len(sizes)-1 and 'terminal_contact_tolerance_m' in action:
                contact = dict(action, contact_tolerance_m=action['terminal_contact_tolerance_m'])
            gap = self.series(regions[offset:offset+size], fine, contact, scale)
            offset += size
            if prev is not None:
                best = np.minimum.accumulate(prev)
                changes = np.r_[True, best[1:] < best[:-1]]
                best_idx = np.maximum.accumulate(np.where(changes, np.arange(len(fine)), 0))
                prior = np.arange(len(fine))-1
                if action.get('direction_policy') == 'contact_window':
                    prior = np.minimum(prior, np.searchsorted(travel, travel-.003, side='right')-1)
                valid = prior >= 0
                parents.append(np.where(valid,best_idx[np.maximum(prior,0)],-1))
                gap = np.maximum(gap,np.where(valid,best[np.maximum(prior,0)],1e6))
            prev = gap
        bins = np.searchsorted(clock, fine)
        np.minimum.at(errors, bins, prev)
        if prev.min() >= 1e6:
            return errors, {}
        index = int(np.argmin(prev))
        chain = [float(fine[index])]
        for parent in reversed(parents):
            index = parent[index]
            chain.append(float(fine[index]))
        detail = {'handle_contact_times': chain[::-1]}
        if action.get('direction_policy') == 'contact_window':
            default_action = dict(action)
            del default_action['direction_policy']
            default_errors, default_detail = self.hinge_path(regions, clock, start, default_action, scale)
            if default_errors.min() < errors.min():
                detail = default_detail
            return np.minimum(errors, default_errors), detail
        return errors, detail

    def held_press(self, anchor, definition, scale):
        """Track the lower surface of the grasped part during one continuous contact."""
        close, release = anchor
        source, target = (s['regions'] for s in definition['milestones'])
        offset = self.centering.get((id(source), close), np.zeros(3))
        initial_pose = self.poses[close]
        local = [np.einsum("vi,ij->vj", np.asarray(points) + offset - self.ps[close], initial_pose[:3, :3])
                 for points in definition['moving_parts']]
        shapes = [convex(points) for points in local]
        tips = np.vstack(local)
        solids = self.solids(target)
        normal = -np.asarray(definition.get('press_axis_world',[0.,0.,-1.]))
        target_heights = np.einsum('vi,i->v', np.vstack([vertices(r) for r in target]), normal)
        mid_height = (target_heights.max() + target_heights.min()) / 2
        radius = np.linalg.norm(tips, axis=1).max()
        band = definition['attachment_uncertainty_m']
        required = definition['press_depth_m']
        errors = np.full(len(self.ps), 1e6)
        witnesses = {}
        peak = approach = bottom = None
        for k in range(close+1, len(self.ps) if release is None else release):
            delta_angle = (self.rotations[k-1].inv() * self.rotations[k]).magnitude()
            travel = np.linalg.norm(self.ps[k]-self.ps[k-1]) + radius*delta_angle
            n = max(1, int(np.ceil(travel / definition['press_sample_step_m'])))
            times = np.linspace(k-1, k, n+1)
            if k > close+1:
                times = times[1:]
            rotations = self.slerp(times).as_matrix()
            for time, R in zip(times, rotations):
                p = self.ps[k-1] + (time-k+1)*(self.ps[k]-self.ps[k-1])
                z = p @ normal + np.einsum("vi,i->v", tips, np.einsum("i,ij->j", normal, R))
                gap = min(self.gap(shape, R, p, solids) for shape in shapes)
                height = float(z.min())
                if gap > 2*band or (z.max()+z.min())/2 < mid_height:
                    peak = approach = bottom = None
                    continue
                if approach is None or height > approach:
                    approach, approach_time = height, float(time)
                depth = 0.
                if gap <= band:
                    if peak is None or height-gap > peak:
                        peak, start = height-gap, float(time)
                    depth = max(0., peak-height)
                    stroke_start = start
                    press_time = float(time)
                    mode = 'follow_through'
                    if bottom is None or height < bottom:
                        bottom, bottom_time = height, float(time)
                        load, load_start = approach-height, approach_time
                else:
                    peak = None
                if bottom is not None:
                    rebound = min(load, height-bottom)
                    if rebound > depth:
                        depth, stroke_start, press_time = rebound, load_start, bottom_time
                        mode = 'load_and_unload'
                if depth <= 1e-9:
                    continue
                err = -scale*np.log(min(1., depth/required))
                if err < errors[k]:
                    errors[k] = err
                    witnesses[k] = dict(press_start_time=stroke_start, press_contact_time=press_time,
                                        press_end_time=float(time), press_depth_m=float(depth), press_mode=mode)
        return errors, witnesses

    def held_contact(self,poses,anchor,definition):
        close,end=anchor
        dst=self.solids(definition['milestones'][1]['regions'])
        pieces=[]
        shift=self.centering.get((id(definition['milestones'][0]['regions']),close),np.zeros(3))
        ref=np.linalg.inv(poses[close])
        for pts in definition['moving_parts']:
            local=np.einsum('ij,kj->ki',ref[:3,:3],np.asarray(pts)+shift)+ref[:3,3]
            pieces.append(convex(local))
        result=np.full(len(poses),1e6)
        closest_at_grasp=min(self.gap(part,poses[close,:3,:3],poses[close,:3,3],dst) for part in pieces)
        for k in range(close+1,len(poses) if end is None else end):
            R=poses[k,:3,:3];p=poses[k,:3,3]
            gap=min(self.gap(part,R,p,dst) for part in pieces)
            band=definition.get('attachment_uncertainty_m',0.)
            result[k]=max(0.,gap-band) if closest_at_grasp-gap>=band else gap
        return result


def support_distance(center, moved, regions):
    """The projected center must be over the receiver; the bottom can settle onto it."""
    from .region import distance
    outcomes=[]
    for r in regions:
        pts=vertices(r)
        low,high=pts[:,2].min(),pts[:,2].max()
        probe=np.array([center[0],center[1],np.clip(moved[:,2].min(),low,high)])
        gap=max(0.,low-center[2],moved[:,2].min()-high)
        outcomes.append(np.hypot(distance([probe],r)[0],gap))
    return min(outcomes)

def cycle_value(trace, regions, close, release, positions):
    if trace is None:
        return float(motion.distance(positions[close],regions)[0]),close
    return trace.grip(regions,close,release)
