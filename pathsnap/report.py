"""Generate a development report from recorded results."""
from pathlib import Path
import csv
import json
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def records(file):
    with open(Path('results')/file) as f: return list(csv.DictReader(f))
def lines(ax,paragraphs):
    ax.axis('off')
    ax.text(0,.98,'\n\n'.join(textwrap.fill(s,82) for s in paragraphs),ha='left',va='top',fontsize=10,
            linespacing=1.45,transform=ax.transAxes)
def page(pdf,title,paras):
    fig,ax=plt.subplots(figsize=(8.3,11.7));fig.subplots_adjust(left=.08,right=.92,top=.90,bottom=.06);fig.suptitle(title,fontsize=19,y=.95)
    lines(ax,paras);pdf.savefig(fig);plt.close(fig)
def report():
    models=records('model_comparison.csv'); inventory=records('inventory.csv')
    trials=records('bayesian_trials.csv'); routes=records('route_comparison.csv')
    cfg=json.loads(Path('results/config.json').read_text())
    dec=json.loads(Path('artifacts/decoder.json').read_text())
    def avg(rows,k):return np.mean([float(r[k]) for r in rows])
    with PdfPages('results/development_report.pdf') as pdf:
        summaries=[]
        for split in ('train','student_test'):
            rows=[r for r in inventory if r['split']==split]
            n=sum(int(r['observations']) for r in rows);miss=sum(int(r['misses']) for r in rows)
            summaries.append(f"{split}: {len(rows)} trips, {n:,} observations. {miss:,} missing reference candidates ({100*miss/n:.2f}%). Maximum recorded gap {max(float(r['max_gap_s']) for r in rows):g} s. Nonfinite values {sum(int(r['missing_values']) for r in rows)}. Disconnected successive reference arcs {sum(int(r['reference_breaks']) for r in rows)}.")
        page(pdf,'PathSnap | Development results',summaries+[
            'The supplied .route files contain ordered traversals. A dynamic program assigns each GPS point to an occurrence along that route with nondecreasing occurrence index and minimum total squared spatial residual. Points may share an arc or skip over short arcs. Parallel arcs and stops can make assignments ambiguous. All point scores use this derived reference.',
            'Projection uses local equirectangular metres centred on each network. All finite, time-ordered track rows are preserved. A spatial index retrieves eight nearest directed segments. Distance, segment length, projected fraction, direction agreement over three time windows, speed and endpoint overshoot are the classifier inputs. IDs only join geometry, predictions and targets.',
            'Five grouped folds keep whole trips together. Each fit uses up to 256 randomly drawn covered observations per training trip and every candidate of each selected point. Held-out scoring includes every observation. The cap limits the influence of very long trajectories. Reported Top-1 and arc macro F1 weight each trip equally. Log loss is conditional on candidate coverage.'])
        names=[r['model'].replace('_',' ') for r in models]
        fig,axes=plt.subplots(1,4,figsize=(11.7,6),sharey=True,layout='constrained')
        for ax,key,title in zip(axes,['cv_macro_f1','top1','macro_f1','log_loss'],['CV macro F1','Test Top-1','Test macro F1','Test log loss']):
            ax.barh(names,[float(r[key]) for r in models],color='#156956');ax.set_title(title)
        axes[0].invert_yaxis();fig.suptitle('Candidate classifier comparison');pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11.7,6));ax.axis('off')
        keys=['cv_macro_f1','top1','macro_f1','log_loss','continuity','score_seconds']
        table=ax.table(cellText=[[r['model']]+[f'{float(r[k]):.4f}' for k in keys] for r in models],
           colLabels=['Model','CV F1','Test Top-1','Test F1','Log loss','Continuity','Score time (s)'],loc='center')
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
        ax.set_title('Mean per-trip comparison | continuity uses successive observation arcs')
        pdf.savefig(fig);plt.close(fig)
        rationale={'logistic':'Tests an additive decision boundary using standardised geometry; minimises regularised logistic loss.',
            'naive_bayes':'Fits class-conditional Gaussian distributions. Correlated distances and heading estimates can distort probabilities.',
            'decision_tree':'Uses Gini splits for nonlinear interactions. Depth and minimum leaf size limit unstable splits.',
            'extra_trees':'Averages randomised trees with Gini splits. Captures interactions at higher scoring cost.',
            'boosting':'Fits successive trees to binary log loss, allowing corrective splits for confusing geometric candidates.'}
        for group in (list(rationale)[:3],list(rationale)[3:]):
            paras=[]
            for name in group:
                r=next(r for r in models if r['model']==name)
                paras.append(f"{name}. {rationale[name]} Configuration: {cfg['models'][name]}. Training Top-1 {float(r['train_top1']):.3f}, grouped CV Top-1 {float(r['cv_top1']):.3f}, student-test Top-1 {float(r['top1']):.3f}, student-test F1 {float(r['macro_f1']):.3f} and log loss {float(r['log_loss']):.3f}. The training-to-CV gap is {float(r['train_top1'])-float(r['cv_top1']):+.3f}.")
            page(pdf,'Model experiments',paras)
        winner=cfg['family'];base=next(r for r in models if r['model']==winner);tuned=models[-1]
        routeparas=[]
        for method in ('classifier','viterbi'):
            rows=[r for r in routes if r['method']==method]
            routeparas.append(f"{method}: Top-1 {avg(rows,'top1'):.4f}, arc macro F1 {avg(rows,'macro_f1'):.4f}, normalised route edit distance {avg(rows,'normalized_edit'):.4f}, continuity between successive route arcs {avg(rows,'route_arc_continuity'):.4f}. Edit distance counts intermediate arcs and fragment breaks.")
        page(pdf,'Model selection and route decoding',[
            f"{winner} led mean trip-level CV Top-1. The Bayesian search tested {len(trials)} configurations of {trials[0]['parameter']} over [{min(cfg['search_space']):.3g}, {max(cfg['search_space']):.3g}]. Three initial trials preceded Gaussian-process expected improvement. Total trial time {sum(float(r['seconds']) for r in trials):.1f} s. Selected overrides: {cfg['selected']}.",
            f"CV Top-1 moved from {float(base['cv_top1']):.7f} to {float(tuned['cv_top1']):.7f}. Student-test Top-1 moved from {float(base['top1']):.4f} to {float(tuned['top1']):.4f}. The selected search did not establish a meaningful accuracy gain.",
            'Viterbi states are directed segment projections. Emissions normalise candidate log odds. Transitions compare directed shortest travel distance with GPS displacement and constrain speed. The decoder maximises cumulative log probability, stores backpointers and recovers traversed connector arcs. It starts a new fragment across a long gap or if no transition is feasible.',
            f"Training-derived settings: 75th percentile reference residual {dec['distance_scale_m']:.2f} m; 99th percentile observed speed {dec['speed_mps']:.2f} m/s; 99.5th percentile time gap {dec['gap_s']:.1f} s. Small backwards motion on the same arc is allowed within one residual scale."]+routeparas)
        fig,ax=plt.subplots(figsize=(9,8));ax.imshow(plt.imread('results/route_overlay.png'));ax.axis('off')
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        page(pdf,'Evaluation interface and remaining evidence',[
            'Laptop 1 discovers the configured trip directories. A FastAPI POST launches a fresh inference worker for the selected trip. After decoding, a separate scoring function reads private reference data and returns aggregate agreement. Laptop 2 requests the catalog and draws the GPS observations and predicted fragments on a shared geographic scale.',
            'The development map shown is a static spatial overlay. The repository includes Folium generation code. The final submission needs a rendered Folium export and screenshots from the live two-laptop demonstration, including the receiving terminal.',
            'This development report was generated with AI assistance from local experiment outputs. The team must review and independently explain the alignment, methodology, evidence and limitations before preparing its final submission.'])
if __name__=='__main__':report()
