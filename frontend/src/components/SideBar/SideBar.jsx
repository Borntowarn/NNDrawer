import React, { useContext } from 'react';
import './SideBar.css'
import { MainContext } from '../context/MainContext';

export default () => {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const {nodeTypes} = useContext(MainContext)
  console.log(nodeTypes)

  return (
    <aside>
      <div className="description">You can drag these nodes to the pane on the right.</div>
      <div>
        {nodeTypes.map((type, i) => (
          <div className="dndnode" onDragStart={(event) => onDragStart(event, type)} draggable key={i}>
            {type}
          </div>
        ))}
      </div>
    </aside>
  );
};