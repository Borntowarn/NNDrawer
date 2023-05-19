import DropNode from './DropNode/DropNode'
import { useScrollbar } from '../../../hooks/use-scrollbar'
import { useRef, useEffect, useContext, useState } from 'react'
import { MainContext } from '../../../context/MainContext'
import './DropDown.css'

export default function DropDown({mode}) {
  const todoWrapper = useRef(null)
  const { projects } = useContext(MainContext)
  const [allowedProjects, setProjects] = useState(projects)
  const [hasScroll, setScroll] = useState(allowedProjects.length > 8)


  const handleChange = (value) => {
    setProjects(projects.filter(elem => elem.name.toLowerCase().includes(value.toLowerCase())))
  }

  useEffect(() => {
    setProjects(projects)
  }, [projects])

  useEffect(() => {
    setScroll(allowedProjects.length > 8)
  }, [allowedProjects])

  useScrollbar(todoWrapper, hasScroll);

  return (
    <div className={mode ? 'dropdown active' : 'dropdown'}>
      <input placeholder='Project title' className='project-seacrch' onChange={(e) => handleChange(e.target.value)} type="text" />
      <div ref={todoWrapper} className={hasScroll ? 'project-list-holder down' : 'project-list-holder'}>
        <div className='project-list'>
          {allowedProjects.map((project, i) => (
            <DropNode title={project.name} data={project} key={i}/>
          ))}
        </div>
      </div>
    </div>
  )
}
