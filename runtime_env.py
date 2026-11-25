"""Event-driven environment used in the Taillard runs. Machine and job labels in a schedule start at one."""
import torch
from torch_geometric.data import Data

class BaseShop:
    """Actual simultaneous-completion event clock; no speculative backfill."""
    def __init__(self,spec):
        self.spec=spec;self.J=spec['jobs'];self.M=spec['machines'];self.t=0
        self.next=[0]*self.J;self.running={};self.schedule=[];self.assigned=set()
        self.F=[0]*self.J;self.E=[0]*self.M
        self.R=list(map(sum,spec['durations']))
        self.Q=[sum(spec['durations'][j][o] for j in range(self.J) for o in range(self.M) if spec['routes'][j][o]==m+1) for m in range(self.M)]
        self.L0=max(self.R+self.Q);self.L=self.L0

    def candidates(self):
        busy={row[0] for row in self.running.values()}
        return [j for j in range(self.J) if j not in busy and self.next[j]<self.M]

    def state(self):
        jobs=self.candidates();assert jobs
        groups={}
        for i,j in enumerate(jobs):
            m=self.spec['routes'][j][self.next[j]]-1
            if m not in self.running:groups.setdefault(m,[]).append(i)
        assert groups,'Automatic clock advancement must leave a dispatch opportunity'
        counts={m:sum(self.spec['routes'][j][self.next[j]]==m+1 for j in jobs) for m in range(self.M)}
        rows=[];machine_ids=[];nodes=[]
        for j in jobs:
            o=self.next[j];m=self.spec['routes'][j][o]-1;p=self.spec['durations'][j][o]
            nextp=self.spec['durations'][j][o+1] if o+1<self.M else 0
            rows.append([p/self.L0,self.R[j]/self.L0,(self.M-o)/self.M,nextp/self.L0,
                         self.Q[m]/self.L0,max(0,self.E[m]-self.t)/self.L0,
                         max(0,self.t-self.F[j])/self.L0,counts[m]/self.J])
            machine_ids.append(m);nodes.append(j*self.M+o+1)
        # A virtual machine context node pools ready-frontier descriptors.
        # Bipartite stars avoid all candidates seeing the identical clique.
        edges=[]
        for m in sorted(set(machine_ids)):
            members=[i for i,required in enumerate(machine_ids) if required==m]
            context=len(rows);rows.append([sum(rows[i][k] for i in members)/len(members) for k in range(8)])
            for i in members:edges.extend([[i,context],[context,i]])
        graph=Data(x=torch.tensor(rows,dtype=torch.float32),edge_index=torch.tensor(edges,dtype=torch.long).T.contiguous())
        return dict(graph=graph,candidate_count=len(jobs),jobs=jobs,nodes=nodes,
                    groups=[dict(machine=m,indices=group) for m,group in sorted(groups.items())],time=self.t,bound=self.L)

    def advance(self,state,choices):
        old=self.L;before=self.t;assert len(choices)==len(state['groups'])
        actual=[]
        for group,choice in zip(state['groups'],choices):
            assert 0<=choice<len(group['indices'])
            i=group['indices'][choice];j=state['jobs'][i];o=self.next[j];m=group['machine'];node=j*self.M+o+1
            assert node not in self.assigned and m not in self.running
            assert self.spec['routes'][j][o]==m+1 and self.F[j]<=self.t
            p=self.spec['durations'][j][o];end=self.t+p
            self.running[m]=(j,o,end);self.assigned.add(node)
            self.F[j]=end;self.E[m]=end;self.R[j]-=p;self.Q[m]-=p
            row=dict(node=node,job=j+1,operation=o+1,machine=m+1,start=self.t,end=end,duration=p)
            self.schedule.append(row);actual.append(row)
        self.L=max([f+r for f,r in zip(self.F,self.R)]+[e+q for e,q in zip(self.E,self.Q)])
        assert self.L>=old
        while True:
            assert self.running
            self.t=min(row[2] for row in self.running.values())
            completed=[m for m,row in self.running.items() if row[2]==self.t]
            for m in completed:
                j,o,end=self.running.pop(m);assert self.next[j]==o;self.next[j]+=1
            done=all(o==self.M for o in self.next)
            if done:break
            if any(self.spec['routes'][j][self.next[j]]-1 not in self.running for j in self.candidates()):break
        if done:assert self.L==self.t and not self.running and len(self.assigned)==self.J*self.M
        return (old-self.L)/self.L0,done,dict(bound_increment=self.L-old,elapsed_time=self.t-before,actual_dispatch=actual)

class Shop(BaseShop):
    def state(self):
        state=super().state()
        extra=[];machine_ids=[]
        for j in state['jobs']:
            o=self.next[j];machine_ids.append(self.spec['routes'][j][o]-1)
            if o+1<self.M:
                following=self.spec['routes'][j][o+1]-1
                extra.append([self.Q[following]/self.L0,
                              max(0,self.E[following]-self.t)/self.L0])
            else: extra.append([0.,0.])
        # Preserve the base's sorted virtual-machine node order and edges.
        for m in sorted(set(machine_ids)):
            members=[i for i,required in enumerate(machine_ids) if required==m]
            extra.append([sum(extra[i][k] for i in members)/len(members) for k in range(2)])
        graph=state['graph']
        assert len(extra)==graph.num_nodes and graph.x.shape[1]==8
        graph.x=torch.cat([graph.x,torch.tensor(extra,dtype=graph.x.dtype)],dim=1)
        return state

def check_schedule(spec,schedule):
    J,M=spec['jobs'],spec['machines'];by={row['node']:row for row in schedule}
    assert len(schedule)==len(by)==J*M and set(by)==set(range(1,J*M+1))
    for node,row in by.items():
        j,o=divmod(node-1,M)
        assert row['job']==j+1 and row['operation']==o+1 and row['machine']==spec['routes'][j][o]
        assert row['end']-row['start']==spec['durations'][j][o] and row['start']>=0
        if o:assert by[node-1]['end']<=row['start']
    for m in range(1,M+1):
        rows=sorted((r['start'],r['end']) for r in schedule if r['machine']==m)
        assert all(a[1]<=b[0] for a,b in zip(rows,rows[1:]))
    return max(r['end'] for r in schedule)
