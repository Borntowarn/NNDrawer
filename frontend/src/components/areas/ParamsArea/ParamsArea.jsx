import { useContext, useState, useEffect, useRef } from 'react'
import { MainContext } from '../../context/MainContext';
import { useScrollbar } from '../../../hooks/use-scrollbar';
import './ParamsArea.css'

export default function ParamsArea() {
  const { currentNode, setCurrentNode, setNodes } = useContext(MainContext);
  const [updatedData, setUpdatedData] = useState()

  const todoWrapper = useRef(null)
  const hasScrollBar = currentNode.data ? Object.keys(currentNode.data).length > 4: false
  const [allowedKeys, setKeys] = useState(currentNode.data ? Object.keys(currentNode.data) : [])

  useEffect(() => {
    setKeys(currentNode.data ? Object.keys(currentNode.data) : [])
  }, [currentNode])

  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        if ((node.id === currentNode.id) && updatedData) {
          console.log('update', updatedData)
          node.data = {
            ...node.data,
            [updatedData[0]]: updatedData[1],
          };
        }
        return node;
      })
    )
    
    if (currentNode.data) {
      currentNode.data[updatedData[0]] = updatedData[1];
      console.log(currentNode.data[updatedData[0]])
    }
    setCurrentNode(currentNode);
  }, [updatedData, setNodes, setCurrentNode]);

  const handleChange = (value) => {
    setKeys(Object.keys(currentNode.data).filter(elem => elem.toLowerCase().includes(value.toLowerCase())))
  }

  useScrollbar(todoWrapper, hasScrollBar)

  return (
    <div className='params-area'>
      <input placeholder='Parameter title' onChange={(e) => handleChange(e.target.value)}  className='nodes-search'/>
      <div ref={todoWrapper} className="params-list">
        <div>
          {currentNode.data ? 
            allowedKeys.map((param, i) => (
              <div key={i} >
                <div className='param'>
                  <label>{param}</label>
                  <input className='param-input' value={currentNode.data[param]} onChange={(evt) => setUpdatedData([param, evt.target.value])}/>
                </div>
              </div>
          )) : <div className='params-empty-title'>Choose node</div>
          }
        </div>
      </div>
    </div>
  )
}
