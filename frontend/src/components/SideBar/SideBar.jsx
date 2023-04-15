import React, { useContext } from 'react';
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
    <aside className='side_area'>
      <div className="description">You can drag these nodes to the pane on the right.</div>
      <div className='blocks_area'>
        {nodeTypes.map((type, i) => (
          <div className="dndnode" onDragStart={(event) => onDragStart(event, type)} draggable key={i}>
            {type}
          </div>
        ))}
      </div>
      <NodeUpdate />
    </aside>
  );
};