import React, { useContext, useState, useEffect, useRef } from 'react'
import { MainContext } from '../context/MainContext';
import './NodeUpdate.css'
import { useScrollbar } from '../../hooks/use-scrollbar';

export default function NodeUpdate() {
  const { currentNode, setCurrentNode, setNodes } = useContext(MainContext);
  const [updatedData, setUpdatedData] = useState()

  const todoWrapper = useRef(null)
  const hasScrollBar = currentNode.data ? Object.keys(currentNode.data).length > 2: false

  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        if ((node.id === currentNode.id) && updatedData) {
          console.log('update', updatedData)
          node.data = {
            ...node.data,
            [updatedData[0]]: updatedData[1],
          };
        }
        return node;
      })
    )
    
    if (currentNode.data) {
      currentNode.data[updatedData[0]] = updatedData[1];
      console.log(currentNode.data[updatedData[0]])
    }
    setCurrentNode(currentNode);
  }, [updatedData, setNodes, setCurrentNode]);

  useScrollbar(todoWrapper, hasScrollBar)

  return (
    <div ref={todoWrapper} className="params-list">
      <div>
        {currentNode.data ? 
          Object.keys(currentNode.data).map((param, i) => (
            <div key={i} >
              <div className='param'>
                <label>{param}</label>
                <input className='param-input' value={currentNode.data[param]} onChange={(evt) => setUpdatedData([param, evt.target.value])}/>
              </div>
            </div>
        )) : <></>
        }
      </div>
    </div>
  
  )
}