"""Laptop 1. Start from the repository root."""
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(title='PathSnap')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=[])
logger=logging.getLogger('uvicorn.error')

def catalog():
    root=Path(os.environ.get('PATHSNAP_TRIPS','Student folder/student_test'))
    return {p.name:p for p in sorted(root.iterdir()) if p.is_dir() and any(p.glob('*.track'))}

@app.get('/trips')
def trips():
    return list(catalog())

@app.post('/predict/{trip_id}')
def match(trip_id: str):
    directory=catalog().get(trip_id)
    if directory is None: raise HTTPException(404,'Unknown trip')
    command=[os.environ.get('PATHSNAP_PYTHON',sys.executable),'-m','pathsnap.serve_worker',
             str(directory),'--artifacts',os.environ.get('PATHSNAP_ARTIFACTS','artifacts')]
    if root:=os.environ.get('PATHSNAP_REFERENCES'):
        command.extend(['--reference-root',root])
    start=perf_counter()
    try:
        output=subprocess.run(command,check=True,capture_output=True,text=True)
        result=json.loads(output.stdout)
    except (subprocess.CalledProcessError,ValueError):
        logger.exception('Inference failed for %s',trip_id)
        raise HTTPException(500,'Trip processing failed') from None
    result['elapsed_seconds']=perf_counter()-start
    logger.info('POST prediction trip=%s observations=%d fragments=%d metrics=%s seconds=%.3f',
                trip_id,len(result['track']),len(result['route']),result['metrics'],result['elapsed_seconds'])
    return result
