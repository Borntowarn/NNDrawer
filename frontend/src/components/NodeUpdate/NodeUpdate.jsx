import React, { useContext, useState, useEffect } from 'react'
import { MainContext } from '../context/MainContext';
import './NodeUpdate.css'

export default function NodeUpdate() {
  const { currentNode, setCurrentNode, setNodes } = useContext(MainContext);
  const [updatedData, setUpdatedData] = useState()

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

  return (
    <div className="params-area">
        {currentNode.data ? 
          Object.keys(currentNode.data).map((param, i) => (
            <div key={i} >
            <label>label: {param}</label>
            <input value={currentNode.data[param]} onChange={(evt) => setUpdatedData([param, evt.target.value])}/>
            </div>
        )) : <></>
        }
    </div>
  )
}