import React, { useContext, useState, useEffect } from 'react'
import { MainContext } from '../context/MainContext';

export default function NodeUpdate() {
  const { currentNode, setCurrentNode, setNodes } = useContext(MainContext);
  const [nodeName, setNodeName] = useState(currentNode.data ? currentNode.data.label : ''); // currentNode ? currentNode.data.label : ''

  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === currentNode.id) {
          node.data = {
            ...node.data,
            label: nodeName,
          };
        }
        return node;
      })
    )
    
    if (currentNode.data) {
      currentNode.data.label = nodeName
    }
    setCurrentNode(currentNode);
  }, [nodeName, setNodes, setCurrentNode]);

  return (
    <div className="updatenode__controls">
        <label>label:</label>
        <input value={currentNode.data ? currentNode.data.label : ''} onChange={(evt) => setNodeName(evt.target.value)} />
    </div>
  )
}