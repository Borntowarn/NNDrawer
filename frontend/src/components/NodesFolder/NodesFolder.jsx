import { useContext } from 'react'
import { MainContext } from '../../context/MainContext'
import '../NodesFolder/NodesFolder.css'

export default function NodesFolder({folder}) {
    const { setShowedNodes } = useContext(MainContext)

    const handleFolderClick = () => {
      setShowedNodes(folder)
    }

  return (
    <div>
        <div className='dndnode' draggable={false} onClick={() => handleFolderClick()}>{folder}</div>
    </div>

  )
}
