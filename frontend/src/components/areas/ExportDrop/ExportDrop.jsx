import { useContext, useState } from 'react'
import { MainContext } from '../../context/MainContext'
import './ExportDrop.css'

export default function ExportDrop({mode}) {
  const { reactFlowInstance, setSavedNodes } = useContext(MainContext)
  const [blockTitle, setBlockTitle] = useState('')
  const handleChange = (value) => {
    setBlockTitle(value)
  }

  const handleClick = () => {
    if (reactFlowInstance) {
        const flow = reactFlowInstance.toObject();

        console.log("FLOW", flow)
        flow.nodes = flow.nodes.filter(nds => nds.id.slice(-5) != '_open')

        const newNode = {
            type: 'customNode1',
            data: { 
                label: blockTitle,
                buttonState: false,
                include: flow
            }
        }
        setSavedNodes((nds) => nds.concat(newNode))
    }
  }

  return (
    <div className={mode ? 'export-drop active' : 'export-drop'}>
      <input placeholder='Block title' className='export-input' onChange={(e) => handleChange(e.target.value)} type="text" />
      <button className='export-create' onClick={() => handleClick()}>create</button>
    </div>
  )
}
