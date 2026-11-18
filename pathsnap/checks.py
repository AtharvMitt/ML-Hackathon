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

        # A long stationary dwell contains many observations but no sampling gap.
        # The subsequent time jump and duplicate timestamp must each start a fragment.
        dwell = np.arange(60)
        restart = dwell[-1] + config["gap_s"] + 1
        times = np.r_[dwell, restart, restart]
        stationary = np.tile((nodes[0] + nodes[1]) / 2, (len(times), 1))
        np.savetxt(copy/'unseen.track', np.column_stack((stationary, times)))
        _, pred, fragments, starts = predict(copy, model, config)
        assert len(pred) == len(times)
        assert starts == [0, len(dwell), len(dwell) + 1]
        for first, end, fragment in zip(starts, starts[1:] + [len(times)], fragments):
            assert set(pred[first:end]).issubset(fragment)
    print('Passed reference removal, ID renumbering, synthetic turns, dwell and time-gap checks.')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trips',type=Path,default=Path('Student folder/student_test'))
    run(p.parse_args().trips)
