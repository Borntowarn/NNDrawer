import { useCallback, useContext } from 'react';
import { MainContext } from '../../../context/MainContext';
import axios from 'axios'
import constants from '../../../constants/constants';

export default function SaveButton() {
  const { reactFlowInstance,
    currentProject,
    setCurrentProject,
    setProjects,
    projects,
    currentTitle,
    authData} = useContext(MainContext);
  const flowKey = 'load-project';

  const onSave = useCallback(() => {
    if (reactFlowInstance) {
      localStorage.setItem(flowKey, currentProject);
      }
  }, [reactFlowInstance, currentProject])

  const updateProjectTitile = () => {
    if (currentProject != currentTitle) {
      if (!(projects.find(elem => elem.name === currentTitle))) {
        setCurrentProject(currentTitle)
      } else {
        console.log('Title already exist')
        return false
      }
    }
    return true
  }

  const handleClick = async () => {
    const flow = reactFlowInstance.toObject();

    const projectTemp = {
      name: currentTitle,
      instance: flow
    }

    console.log(flow)
    onSave(flow)
    
    if (currentTitle && updateProjectTitile()) {
      console.log("project: ", currentProject)

      if (projects.find(prj => prj.name == currentProject)) {
        setProjects(projects.map((prj) =>
        prj.name === currentProject ? 
        {...prj, name: currentTitle, instance: flow } : prj ))
        console.log('UPDATE PROJECT: ', currentProject)

      } else {
        setProjects([...projects, projectTemp])
        console.log('CREATE PROJECT: ', projectTemp)
      }
      
      try {
          axios.post(constants.urls.add_project, 
          {
            idUser: authData,
            data: JSON.stringify(projectTemp),
            headers: {'Content-Type': 'application/json'},
            withCredentials: true,
          })
      } catch(err) {
        console.log("ERROR: ", err) }
    } else {
      console.log("Enter title of your project")
    }
  }
  return (
    <button onClick={() => handleClick()}>Save</button>
  )
}
