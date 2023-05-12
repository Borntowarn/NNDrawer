import { useContext, useState, useEffect, useRef } from 'react'
import { MainContext } from '../../../context/MainContext';
import { useScrollbar } from '../../../hooks/use-scrollbar';
import './ParamsArea.css'

export default function ParamsArea() {
  const { currentNode, setCurrentNode, setNodes, nodes } = useContext(MainContext);
  const [updatedData, setUpdatedData] = useState()

  const todoWrapper = useRef(null)
  const hasScrollBar = currentNode.data ? Object.keys(currentNode.data.Args).length > 4: false
  const [allowedKeys, setKeys] = useState(currentNode.data ? Object.keys(currentNode.data.Args) : [])

  useEffect(() => {
    setKeys(currentNode.data ? Object.keys(currentNode.data.Args) : [])
  }, [currentNode])

  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        if ((node.id === currentNode.id) && updatedData) {
          console.log('update', updatedData)
          node.data.Args = {
            ...node.data.Args,
            [updatedData[0]]: updatedData[1],
          };
        }
        return node;
      })
    )
    
    if (currentNode.data) {
      currentNode.data.Args[updatedData[0]] = updatedData[1];
      console.log(currentNode.data.Args[updatedData[0]])
    }
    setCurrentNode(currentNode);
  }, [updatedData, setNodes, setCurrentNode]);

  const handleChange = (value) => {
    setKeys(Object.keys(currentNode.data.Args).filter(elem => elem.toLowerCase().includes(value.toLowerCase())))
  }

  useScrollbar(todoWrapper, hasScrollBar)

  console.log(nodes)
  console.log(currentNode)

  return (
    <div className='params-area'>
      <input placeholder='Parameter title' onChange={(e) => handleChange(e.target.value)}  className='nodes-search'/>
      <div ref={todoWrapper} className="params-list">
        <div>
          {currentNode.data ? 
            allowedKeys.map((param, i) => (
              <div key={i} >
                <div className='param'>
                  <label>Title: {param} 
                    {currentNode.data.Args[param] ? 
                    ' Type: ' + currentNode.data.Args[param].Type 
                    + ' Default: ' + currentNode.data.Args[param].Default : ''}
                  </label>
                  <input className='param-input' value={
                    currentNode.data.Args[param] ? 
                      typeof(currentNode.data.Args[param]) === 'string' ?
                       currentNode.data.Args[param] : "" + currentNode.data.Args[param].Default
                      : ''
                  } onChange={(evt) => setUpdatedData([param, evt.target.value])}/>
                </div>
              </div>
          )) : <div className='params-empty-title'>Choose node</div>
          }
        </div>
      </div>
    </div>
  )
}
