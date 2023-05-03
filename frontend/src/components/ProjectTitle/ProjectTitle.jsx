import React, { useContext, useState } from 'react'
import { MainContext } from '../context/MainContext'
import './ProjectTitle.css'

export default function ProjectTitle() {
  const {
    currentProject,
    setCurrentProject,
    projects,
    setProjects
  } = useContext(MainContext)

  const handleChange = (value) => {
    setProjects(projects.map((prj) =>
      prj.name === currentProject ? 
      {...prj, name: value} : prj
    ))
    setCurrentProject(value)
  }

  return (
    <input className='title-input' onChange={(e) => handleChange(e.target.value)} placeholder='project title' 
    value={currentProject ? currentProject : ''}/>
  )
}
