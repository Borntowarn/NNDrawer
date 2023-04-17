import { useContext, useRef, useState } from 'react';
import { MainContext } from '../../context/MainContext';
import { useScrollbar } from '../../../hooks/use-scrollbar';
import './NodesArea.css'

export default function NodesArea() {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const {nodeTypes} = useContext(MainContext)
  const types = Object.keys(nodeTypes)

  const todoWrapper = useRef(null)
  const hasScroll = types.length > 5
  const [allowedBlocks, setBlocks] = useState(types)

  const handleChange = (value) => {
    setBlocks(types.filter(elem => elem.includes(value)))
  }

  useScrollbar(todoWrapper, hasScroll);

  return (
    <div className='dnd-area'>
      <input onChange={(e) => handleChange(e.target.value)} placeholder='Block title' className='nodes-search'/>
      <div ref={todoWrapper} className='nodes-list'>
        <div>
          {allowedBlocks.map((type, i) => (
            <div className="dndnode" onDragStart={(event) => onDragStart(event, type)} draggable key={i}>
              {type}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
