import { useContext } from 'react'
import { MainContext } from '../../context/MainContext'
import '../NodesFolder/NodesFolder.css'

export default function NodesFolder({folder}) {
    const { setShowedNodes, showedNodes } = useContext(MainContext)

    const handleFolderClick = () => {
      console.log('path was changed')

      if (showedNodes.length == 0) {
        setShowedNodes([...showedNodes, folder])
        console.log([...showedNodes, folder])
      } else {
        setShowedNodes([...showedNodes, ...[folder, 'Classes']])
        console.log([...showedNodes, ...[folder, 'Classes']])
      }


    }

  return (
    <div>
        <div className='dndnode' draggable={false} onClick={() => handleFolderClick()}>{folder}</div>
    </div>

  )
}
