import { useState, useCallback, useContext } from 'react';
import { MainContext } from '../context/MainContext';


export default function SaveButton() {
  const { reactFlowInstance } = useContext(MainContext);
  const flowKey = 'load-flow';

  const onSave = useCallback(() => {
    if (reactFlowInstance) {
        const flow = reactFlowInstance.toObject();
        localStorage.setItem(flowKey, JSON.stringify(flow));
    }
  }, [reactFlowInstance])

  return (
    <button onClick={() => onSave()}>Save</button>
  )
}
