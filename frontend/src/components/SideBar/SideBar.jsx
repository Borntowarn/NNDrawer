import React, { useContext, useRef } from 'react';
import './SideBar.css'
import { MainContext } from '../context/MainContext';
import NodeUpdate from '../NodeUpdate/NodeUpdate';

import { useScrollbar } from '../../hooks/use-scrollbar';

export default () => {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const {nodeTypes} = useContext(MainContext)

  const todoWrapper = useRef(null)
  const hasScroll = nodeTypes.length > 5

  useScrollbar(todoWrapper, hasScroll);

  return (
    <aside className='side-area'>
      <div className='dnd-area'>
        <input className='nodes-search'/>
          <div className='nodes-list' ref={todoWrapper}>
            {nodeTypes.map((type, i) => (
                <div className="dndnode" onDragStart={(event) => onDragStart(event, type)} draggable key={i}>
                  {type}
                </div>
              ))}
            
          </div>
      </div>
      <div className='update-area'>
        <div className='params-title'>Params</div>
        <NodeUpdate />
      </div>
    </aside>
  );
};