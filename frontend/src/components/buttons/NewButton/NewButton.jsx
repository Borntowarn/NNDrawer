import { useContext } from 'react'
import { MainContext } from '../../../context/MainContext'

export default function NewButton() {
  const { createNewProject } = useContext(MainContext)

  return (
    <button onClick={() => createNewProject()}>New</button>
  )
}
