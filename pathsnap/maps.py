"""Development-only overlays. Keep reference layers out of the evaluator UI."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .predict import predict
from .reference import read_reference

def render(root, interactive):
    # Median observation count gives a reproducible, score-independent example.
    trips=sorted((p for p in root.iterdir() if p.is_dir()),key=lambda p:len(next(p.glob('*.track')).read_bytes().splitlines()))
    directory=trips[len(trips)//2]
    net,_,fragments,_=predict(directory,joblib.load('artifacts/model.joblib'),
                               json.loads(Path('artifacts/decoder.json').read_text()))
    ref=read_reference(directory,net)
    gps=net.track[:,:2]
    roads=net.nodes[net.arcs]
    lo,hi=gps.min(axis=0),gps.max(axis=0)
    pad=np.maximum(.04*(hi-lo),1e-5)
    mask=((roads.max(axis=1)>=lo-pad)&(roads.min(axis=1)<=hi+pad)).all(axis=1)
    visible=roads[mask]
    layers=[('Road network',visible,'#a8b1bb',.65),
            ('Reference',roads[ref],'#e69530',4),
            ('Prediction',roads[np.concatenate(fragments)],'#00725f',1.6)]
    if interactive:
        import folium
        # Fractional zoom lets the initial view fit the trip closely.
        m=folium.Map(tiles=None,prefer_canvas=True,zoom_snap=.25)
        # A faded keyless basemap keeps the grey network and route colours legible.
        folium.TileLayer('OpenStreetMap',name='Basemap (faded)',opacity=.45).add_to(m)
        for name,segments,color,width in layers:
            group=folium.FeatureGroup(name=name).add_to(m)
            folium.PolyLine(segments[:,:,::-1].tolist(),color=color,weight=max(1,width)).add_to(group)
            if name=='Road network':
                # GPS sits above the network but below the routes, so off-road noise stays
                # visible without hiding the reference and prediction lines.
                gps_group=folium.FeatureGroup(name='GPS observations').add_to(m)
                for lon,lat in gps:
                    folium.CircleMarker([lat,lon],radius=2,stroke=False,fill=True,fill_color='#59347c',
                                        fill_opacity=.55).add_to(gps_group)
        folium.LayerControl(collapsed=False).add_to(m)
        # Static legend naming the trip, partition and each layer's styling for the report figure.
        rows=''.join(f'<div><span style="display:inline-block;width:22px;height:0;border-top:{max(2,round(width))}px solid {color};vertical-align:middle;margin-right:6px"></span>{name}</div>'
                     for name,_,color,width in layers)
        rows+='<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#59347c;margin:0 11px 0 7px"></span>GPS observations</div>'
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;bottom:28px;right:12px;z-index:9999;background:white;padding:8px 10px;'
            'border:1px solid #888;border-radius:4px;font:13px sans-serif;line-height:1.5">'
            f'<b>Trip {directory.name}</b> · {root.name} partition (development)<br>{rows}</div>'))
        m.fit_bounds([lo[::-1].tolist(),hi[::-1].tolist()])
        m.save('results/development_map.html')
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    for name,edges,color,width in layers:
        ax.add_collection(LineCollection(edges,linewidths=width,colors=color,label=name))
    ax.scatter(*gps.T,s=2,c='#59347c',label='GPS observations')
    ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),
           xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title=f'{root.name} trip {directory.name}')
    ax.set_aspect(1/np.cos(np.deg2rad(gps[:,1].mean())))
    ax.legend(loc='best');fig.savefig('results/route_overlay.png',dpi=160);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--partition',type=Path,default=Path('Student folder/student_test'))
    p.add_argument('--folium',action='store_true')
    args=p.parse_args()
    render(args.partition,args.folium)
