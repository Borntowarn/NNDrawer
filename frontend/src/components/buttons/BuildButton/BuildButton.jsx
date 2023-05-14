import React from 'react'
import axios from 'axios'
import constants from '../../../constants/constants'
import { useContext } from 'react'
import { MainContext } from '../../../context/MainContext'

export default function BuildButton() {
    const { currentProject, projects } = useContext(MainContext)
  
  const onBuild = async (e) => {
    console.log("BUILD: ", currentProject, projects.find(elem => elem.name === currentProject))
    console.log("BUILD: ", projects.find(elem => elem.name === currentProject))
    try {
        const response = await axios.post(constants.urls.build,
                projects.find(elem => elem.name === currentProject)
        )
        console.log("RESPONSE: ", response)
    } catch(err) { 
        console.log("ERROR", err)
    }
  }

  return (
    <button onClick={() => onBuild()}>Buld</button>
  )
}
