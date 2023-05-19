import React, { useContext, useEffect, useState } from 'react'
import { MainContext } from '../../context/MainContext'
import './ProjectTitle.css'

export default function ProjectTitle() {
  const {
    currentProject,
    setCurrentTitle,
  } = useContext(MainContext)

  const [inputValue, setInputValue] = useState(currentProject ? currentProject : '')

  useEffect(() => {
    setInputValue(currentProject ? currentProject : '')
    setCurrentTitle(currentProject)
  }, [currentProject])

  const handleChange = (value) => {
    setInputValue(value)
    setCurrentTitle(value)
  }
  return (
    <input className='title-input' placeholder='project title' onChange={(e) => handleChange(e.target.value)}
    value={inputValue}/>
  )
}
