import React, { useContext, useEffect, useState } from 'react'
import { MainContext } from '../../context/MainContext'
import './ProjectTitle.css'

export default function ProjectTitle() {
  const {
    currentProject,
    setCurrentProject,
    projects,
    setProjects
  } = useContext(MainContext)

  const [inputValue, setInputValue] = useState(currentProject ? currentProject : '')

  useEffect(() => {
    setInputValue(currentProject ? currentProject : '')
  }, [currentProject])

  const handleChange = (value) => {
    setInputValue(value)
  }

  const handleKeyChange = (e) => {
    const value = e.target.value
    if(e.key === 'Enter') {
      
      if (projects.find(elem => elem.name === value)) {
        console.log('Project with the same title already exist')
      } else {
        console.log(currentProject)
        setProjects(projects.map((prj) =>
        prj.name === currentProject ? 
        {...prj, name: value} : prj
        ))
        setCurrentProject(value)        
      }
    }
  }

  return (
    // <input className='title-input' onChange={(e) => handleChange(e.target.value)} placeholder='project title' 
    // value={currentProject ? currentProject : ''}/>

    <input className='title-input' onKeyDown={(e) => handleKeyChange(e)} placeholder='project title'
    onChange={(e) => handleChange(e.target.value)}
    value={inputValue}/>
  )
}
