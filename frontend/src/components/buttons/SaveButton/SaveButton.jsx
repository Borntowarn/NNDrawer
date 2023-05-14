import { useCallback, useContext } from 'react';
import { MainContext } from '../../../context/MainContext';
import axios from 'axios'
import constants from '../../../constants/constants';

export default function SaveButton() {
  const { reactFlowInstance,
    currentProject,
    setProjects,
    projects,
    authData} = useContext(MainContext);
  const flowKey = 'load-project';

  const onSave = useCallback(() => {
    if (reactFlowInstance) {
      localStorage.setItem(flowKey, currentProject);
      }
  }, [reactFlowInstance, currentProject])

  const handleClick = async () => {
    const flow = reactFlowInstance.toObject();
    console.log(flow)
    onSave(flow)
    if (currentProject) {
      if (projects.find(prj => prj.name == currentProject)) {
        setProjects(projects.map((prj) =>
        prj.name === currentProject ? 
        {...prj, instance: flow } : prj ))
      } else {
        setProjects([...projects, {
          name: currentProject,
          instance: flow
        }])
      }
      console.log('PROJECTS: ', projects)
      
      try {
        const response = await axios.post(constants.urls.add_project, 
            {
                idUser: '1',
                data: JSON.stringify(projects.find(prj => prj.name == currentProject)),
                headers: {'Content-Type': 'application/json'},
                withCredentials: true,
            })
            console.log(JSON.stringify(response))
      } catch(err) {
        console.log("ERROR: ", err) }
    } else {
      console.log("TITLE_ERROR: enter title of your project")
    }
  }
  return (
    <button onClick={() => handleClick()}>Save</button>
  )
}
