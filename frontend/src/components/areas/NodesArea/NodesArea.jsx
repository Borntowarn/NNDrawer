import { useContext, useEffect, useRef, useState } from 'react';
import { MainContext } from '../../../context/MainContext';
import { useScrollbar } from '../../../hooks/use-scrollbar';
import './NodesArea.css'

export default function NodesArea() {
  const onDragStart = (event, newNode) => {
    const data = JSON.stringify(newNode);
    event.dataTransfer.setData('node', data);
    event.dataTransfer.effectAllowed = 'move';
  };
  
  const {savedNodes} = useContext(MainContext)

  useEffect(() => {
    setNodes(savedNodes)
  }, [savedNodes])

  const todoWrapper = useRef(null)
  const hasScroll = savedNodes.length > 5
  const [allowedNodes, setNodes] = useState(savedNodes)

  const handleChange = (value) => {
    setNodes(savedNodes.filter(elem => elem.data.label.toLowerCase().includes(value.toLowerCase())))
  }

  useScrollbar(todoWrapper, hasScroll);

  return (
    <div className='dnd-area'>
      <input onChange={(e) => handleChange(e.target.value)} placeholder='Node title' className='nodes-search'/>
      <div ref={todoWrapper} className='nodes-list'>
        <div>
          {allowedNodes.map((node, i) => (
            <div className="dndnode" onDragStart={(event) => onDragStart(event, node)} draggable key={i}>
              {node.data.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
