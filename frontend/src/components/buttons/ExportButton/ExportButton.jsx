import { useContext } from "react"
import { MainContext } from "../../context/MainContext"

export default function ExportButton() {
  const { reactFlowInstance, setSavedNodes } = useContext(MainContext)
  
  const onExport = () => {
    if (reactFlowInstance) {
        const flow = reactFlowInstance.toObject();
        const newNode = {
            type: 'customNode1',
            data: { 
                label: 'export_node', 
                include: flow
            }
        }
        setSavedNodes((nds) => nds.concat(newNode))
    }
  }

  return (
    <button onClick={onExport}>ExportButton</button>
  )
}
