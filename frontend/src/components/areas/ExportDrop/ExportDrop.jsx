import { useContext, useState } from 'react'
import { MainContext } from '../../../context/MainContext'
import constants from '../../../constants/constants'
import axios from 'axios'
import './ExportDrop.css'

export default function ExportDrop({mode}) {
  const { reactFlowInstance, setCustomNodes,authData } = useContext(MainContext)
  const [blockTitle, setBlockTitle] = useState('')
  const handleChange = (value) => {
    setBlockTitle(value)
  }

  const handleExportClick = () => {
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
        // try {
        //   const response = axios.post(constants.urls.blocks, 
        //   {
        //     idUser: authData,
        //     data: JSON.stringify(newNode),
        //     headers: {'Content-Type': 'application/json'},
        //     withCredentials: true,
        //   })
        //   console.log(response)
        // } catch(err) {
        //   console.log("ERROR: ", err)
        // }

        setCustomNodes((nds) => nds.concat(newNode))
    }
  }

  return (
    <div className={mode ? 'export-drop active' : 'export-drop'}>
      <input placeholder='Block title' className='export-input' onChange={(e) => handleChange(e.target.value)} type="text" />
      <button className='export-create' onClick={() => handleExportClick()}>create</button>
    </div>
  )
}
