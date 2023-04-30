import DropDown from '../../areas/Dropdown/DropDown'
import { useState } from 'react'

export default function OpenButton() {
  const [active, setActive] = useState(false)
  const testProjects = ['1', '2', '3']

  return (
    <div>
      <button onClick={() => setActive(!active)}>Open</button>
      <DropDown mode={active} data={testProjects}/>
    </div>
  )
}
