"""Exercise the saved predictor using unseen IDs and inputs without reference files."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
import joblib
import numpy as np
from .predict import predict

def run(root):
    model=joblib.load('artifacts/model.joblib')
    config=json.loads(Path('artifacts/decoder.json').read_text())
    source=min((p for p in root.iterdir() if p.is_dir()),key=lambda p:next(p.glob('*.track')).stat().st_size)
    with tempfile.TemporaryDirectory() as temp:
        copy=Path(temp)/'anonymous'
        copy.mkdir()
        for ext in ('track','arcs','nodes'):
            shutil.copyfile(next(source.glob('*.'+ext)),copy/('unseen.'+ext))
        net,base,segments,starts=predict(source,model,config)
        _,anonymous,other,other_starts=predict(copy,model,config)
        assert np.array_equal(base,anonymous) and segments==other and starts==other_starts
        rng=np.random.default_rng(2026)
        nodes=rng.permutation(len(net.nodes))
        inverse=np.argsort(nodes)
        arcs=rng.permutation(len(net.arcs))
        np.savetxt(copy/'unseen.nodes',net.nodes[nodes],fmt='%.12f')
        np.savetxt(copy/'unseen.arcs',inverse[net.arcs[arcs]],fmt='%d')
        _,renumbered,other,other_starts=predict(copy,model,config)
        assert np.array_equal(base,arcs[renumbered]) and other_starts==starts
        assert [arcs[s].tolist() for s in other]==segments

        # An invented three-arm junction; swap only the observed exit.
        nodes=np.array([[-.001,0.],[0.,0.],[.001,0.],[0.,.001]])
        edges=np.array([[0,1],[1,2],[1,3],[1,0],[2,1],[3,1]])
        np.savetxt(copy/'unseen.nodes',nodes)
        np.savetxt(copy/'unseen.arcs',edges,fmt='%d')
        for destination in (2,3):
            gps=np.vstack([np.linspace(nodes[0],nodes[1],8,endpoint=False),
                           np.linspace(nodes[1],nodes[destination],9)])
            np.savetxt(copy/'unseen.track',np.column_stack((gps,np.arange(len(gps)))))
            _,pred,_,_=predict(copy,model,config)
            assert np.array_equal(edges[pred[-1]], [1,destination])
    print('Passed reference removal, trip renaming, node/arc renumbering and synthetic turn checks.')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trips',type=Path,default=Path('Student folder/student_test'))
    run(p.parse_args().trips)
