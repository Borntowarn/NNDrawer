import { useContext } from 'react'
import { MainContext } from '../../context/MainContext'
import '../NodesFolder/NodesFolder.css'

export default function NodesFolder({folder}) {
    const { setShowedNodes, showedNodes } = useContext(MainContext)

    const handleFolderClick = () => {
      if (showedNodes.length == 0) {
        setShowedNodes([...showedNodes, folder])
      } else {
        setShowedNodes([...showedNodes, ...[folder, 'Classes']])
      }
    }

  return (
    <div>
        <div className='dndnode' draggable={false} onClick={() => handleFolderClick()}>{folder}</div>
    </div>

  )
}
