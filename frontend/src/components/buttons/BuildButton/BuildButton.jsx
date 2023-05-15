import React from 'react'
import axios from 'axios'
import constants from '../../../constants/constants'
import { useContext } from 'react'
import { MainContext } from '../../../context/MainContext'

export default function BuildButton() {
    const { currentProject, projects } = useContext(MainContext)  
  
  const onBuild = async (e) => {
    console.log("BUILD: ", currentProject)
    const tempProject = projects.find(elem => elem.name === currentProject)

    if (tempProject) {
      try {
        axios({
          url: constants.urls.build,
          method: 'post',
          responseType: 'blob',
          data: tempProject
        }).then((response) => {
          const href = URL.createObjectURL(response.data);
      
          const link = document.createElement('a');
          link.href = href;
          link.setAttribute('download', 'models.py'); 
          document.body.appendChild(link);
          link.click();
      
          document.body.removeChild(link);
          URL.revokeObjectURL(href);
        });
      } catch(err) { 
        console.log("ERROR", err)
      }
    } else {
      console.log("Can't build project (project doesn't exist)")
    }
  }

  return (
    <button onClick={() => onBuild()}>Buld</button>
  )
}
