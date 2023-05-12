import { useContext, useState } from 'react'
import NodesArea from '../areas/NodesArea/NodesArea'
import constants from '../../constants/constants'
import '../NodesFolder/NodesFolder.css'
import { MainContext } from '../../context/MainContext'

export default function NodesFolder({folder}) {
    const [active, setActive] = useState(false)
    const { setShowedNodes } = useContext(MainContext)

    const handleFolderClick = () => {
      setShowedNodes(folder)
    }

  return (
    <div>
        <div className='dndnode' onClick={() => handleFolderClick()}>{folder}</div>
    </div>

  )
}
