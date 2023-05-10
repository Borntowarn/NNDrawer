import DropDown from '../../areas/Dropdown/DropDown'
import { useState } from 'react'

export default function OpenButton() {
  const [active, setActive] = useState(false)
  const testProjects = ['project_1', 'project_2', 'project_3']

  return (
    <div>
      <button onClick={() => setActive(!active)}>Open</button>
      <DropDown mode={active} data={testProjects}/>
    </div>
  )
}
