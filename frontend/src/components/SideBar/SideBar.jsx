import React, { useContext } from 'react';
import { OverlayScrollbarsComponent } from "overlayscrollbars-react";
import './SideBar.css'
import { MainContext } from '../context/MainContext';
import NodeUpdate from '../NodeUpdate/NodeUpdate';

export default () => {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const {nodeTypes} = useContext(MainContext)
  console.log(nodeTypes)

  return (
    <aside className='side-area'>
      <div className='dnd-area'>
        <input className='nodes-search'/>
        <div className='nodes-list'>
          <OverlayScrollbarsComponent defer>
          {nodeTypes.map((type, i) => (
              <div className="dndnode" onDragStart={(event) => onDragStart(event, type)} draggable key={i}>
                {type}
              </div>
            ))}
          </OverlayScrollbarsComponent>
        </div>
      </div>
      <div className='update-area'>
        <div className='params-title'>Params</div>
        <NodeUpdate />
      </div>
    </aside>
  );
};