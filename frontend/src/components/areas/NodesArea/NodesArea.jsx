import { useContext, useRef } from 'react';
import { MainContext } from '../../context/MainContext';
import { useScrollbar } from '../../../hooks/use-scrollbar';
import './NodesArea.css'

export default function NodesArea() {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const {nodeTypes} = useContext(MainContext)
  const todoWrapper = useRef(null)
  const hasScroll = nodeTypes.length > 5

  useScrollbar(todoWrapper, hasScroll);

  return (
    <div className='dnd-area'>
      <input placeholder='Block title' className='nodes-search'/>
        <div className='nodes-list' ref={todoWrapper}>
          {nodeTypes.map((type, i) => (
              <div className="dndnode" onDragStart={(event) => onDragStart(event, type)} draggable key={i}>
                {type}
              </div>
            ))}
        </div>
    </div>
  )
}
