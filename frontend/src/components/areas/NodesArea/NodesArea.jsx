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

  const getCurrentNodes = (obj, path) => {
    if (path.at(-1) === 'custom'){
      return customNodes
    }
    for (let i=0; i < path.length; i++) {
      obj=obj[path[i]]
    }
    return obj
  }

  const [activeNodes, setActiveNodes] = useState(Object.keys(getCurrentNodes(constants.nodes, showedNodes)))
  const [allowedNodes, setAllowedNodes] = useState(activeNodes)
  const [folderMode, setFolderMode] = useState()


  useEffect(() => {
    updateInstance()
  }, [ showedNodes, customNodes ])

  const todoWrapper = useRef(null)
  const hasScroll = allowedNodes.length > 5

  const updateInstance = () => {
    if (showedNodes.at(-1) === 'custom') {
      setActiveNodes(customNodes)
      setAllowedNodes(customNodes)
    } else {
      setActiveNodes(Object.keys(getCurrentNodes(constants.nodes, showedNodes)))
      setAllowedNodes(Object.keys(getCurrentNodes(constants.nodes, showedNodes)))
    }
    setFolderMode(
      showedNodes.at(-1) != 'custom'  &&
      showedNodes.at(-1) != 'Classes' &&
      !Object.keys(getCurrentNodes(constants.nodes, showedNodes)
      [Object.keys(getCurrentNodes(constants.nodes, showedNodes))[0]])
      .find(elem => (elem === 'Functions' || elem === 'Args'))
    )
  }

  const handleChange = (value) => {
    setAllowedNodes(activeNodes.filter(elem => elem.toLowerCase().includes(value.toLowerCase())))
  }

  const handleFolderBack = () => {
    if (showedNodes.at(-1) === 'Classes') {
      setShowedNodes(showedNodes.slice(0, -2))
      console.log(showedNodes.slice(0, -2))
    } else {
      setShowedNodes(showedNodes.slice(0, -1))
    }
    setActiveNodes(Object.keys(getCurrentNodes(constants.nodes, [])))
    setAllowedNodes(Object.keys(getCurrentNodes(constants.nodes, [])))
  }


  const getObjectArgs = (obj) => {
    if (!!Object.keys(obj).find(elem => elem === 'Functions')) {
      return obj.Functions.__init__ ? obj.Functions.__init__ :  {Args: []}
    } else if (!!Object.keys(obj).find(elem => elem === 'Args')) { 
      return obj
    } else {
      return {Args: []}
    }
  }

  useScrollbar(todoWrapper, hasScroll);

  return (
    <div className='dnd-area'>
      <input onChange={(e) => handleChange(e.target.value)} placeholder='Node title' className='nodes-search'/>
      {showedNodes.length === 0 ? <></> : <button className='folder-back-button' onClick={() => handleFolderBack()}>close</button>}
      <div ref={todoWrapper} className='nodes-list'>
        {folderMode ? 
          <div>
            {allowedNodes.map((folder, i) => ( 
              <NodesFolder folder={folder} key={i}/>
            ))}
          </div> :
          <div>
            {allowedNodes.map((node, i) => (
              <div className="dndnode" onDragStart={(event) => onDragStart(
                event,
                showedNodes.at(-1) === 'custom' && node.data ? node.data.label : node,
                showedNodes.at(-1) === 'custom' ? node : getObjectArgs(getCurrentNodes(constants.nodes, [...showedNodes, node])),

                )} draggable key={i}>
                {showedNodes.at(-1) === 'custom' && node.data ? node.data.label : node}
              </div>
            ))}
          </div>
        }
      </div>
    </div>
  )
}
