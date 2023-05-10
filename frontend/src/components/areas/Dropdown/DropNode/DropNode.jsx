import { useContext } from 'react'
import { MainContext } from '../../../context/MainContext'
import './DropNode.css'

export default function DropNode({title, data}) {
  const { updateInstance } = useContext(MainContext)

  const handleClick = () => {
    updateInstance(data)
  }

  return (
    <div onClick={() => handleClick()} className='dropnode'>{title}</div>
  )
}
