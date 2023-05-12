import { useContext, useEffect, useRef, useState } from 'react';
import { MainContext } from '../../../context/MainContext';
import { useScrollbar } from '../../../hooks/use-scrollbar';
import constants from '../../../constants/constants';
import NodesFolder from '../../NodesFolder/NodesFolder';
import './NodesArea.css'


export default function NodesArea() {
  const onDragStart = (event, title, params) => {
    const data = JSON.stringify({title, params});
    event.dataTransfer.setData('node', data);
    event.dataTransfer.effectAllowed = 'move';
  };
  
  const { showedNodes, setShowedNodes, customNodes, setCustomNodes} = useContext(MainContext)
  const [ activeNodes, setActiveNodes ] = useState(
    showedNodes === 'folders' ? 
    Object.keys(constants.nodes) : constants.nodes[showedNodes]
  )
  const [allowedNodes, setAllowedNodes] = useState(activeNodes)

  useEffect(() => {
    console.log("EFFECT: ", showedNodes)
    if (showedNodes === 'folders') {
      setActiveNodes(Object.keys(constants.nodes))
      setAllowedNodes(Object.keys(constants.nodes))
    }
    else if (showedNodes === 'custom') {
      setActiveNodes(customNodes)
      setAllowedNodes(customNodes)
    }
    else {
      setActiveNodes(Object.keys(constants.nodes[showedNodes]))
      setAllowedNodes(Object.keys(constants.nodes[showedNodes]))
    }
  }, [ showedNodes, customNodes ])

  const todoWrapper = useRef(null)
  const hasScroll = allowedNodes.length > 5

  const handleChange = (value) => {
    setAllowedNodes(activeNodes.filter(elem => elem.toLowerCase().includes(value.toLowerCase())))
  }

  const handleFolderBack = () => {
    setShowedNodes('folders')
    setActiveNodes(Object.keys(constants.nodes))
    setAllowedNodes(Object.keys(constants.nodes))
    console.log('CHANGE SHOWED')
  }

  useScrollbar(todoWrapper, hasScroll);

  return (
    <div className='dnd-area'>
      <input onChange={(e) => handleChange(e.target.value)} placeholder='Node title' className='nodes-search'/>
      {showedNodes === 'folders' ?  <></> : <button className='folder-back-button' onClick={() => handleFolderBack()}>close</button>}
      <div ref={todoWrapper} className='nodes-list'>
        {showedNodes === 'folders' ? 
          <div>
            {allowedNodes.map((folder, i) => (
            <NodesFolder folder={folder} key={i}/>
            ))}
          <NodesFolder folder={'custom'}/>
          </div> :
          <div>
            {allowedNodes.map((node, i) => (
              <div className="dndnode" onDragStart={(event) => onDragStart(
                event,
                showedNodes === 'custom' && node.data ? node.data.label : node,
                showedNodes === 'custom' ? node : constants.nodes[showedNodes][node],
                )} draggable key={i}>
                {showedNodes === 'custom' && node.data ? node.data.label : node}
              </div>
            ))}
          </div>
        }
      </div>
    </div>
  )
}
