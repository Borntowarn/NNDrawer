import { useCallback, useContext } from 'react';
import { MainContext } from '../../context/MainContext';
import axios from 'axios'

export default function SaveButton() {
  const { reactFlowInstance,
    currentProject,
    setProjects,
    projects,
    authData} = useContext(MainContext);
  const flowKey = 'load-project';

  // Just for test
  const PROJECTS_URL = '/projects'

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
      
      // place for axios request
      try {
        const response = await axios.post(PROJECTS_URL, 
            JSON.stringify({
                name: authData.email,
                project: {
                  name: currentProject,
                  instance: flow,
                } 
            }),
            {
                headers: {'Content-Type': 'application/json'},
                withCredentials: true,
            })
            console.log(response.data)
            console.log(response.accessToken)
            console.log(JSON.stringify(response))
    } catch(err) {
        console.log("ERROR: ", err) }
    } else {
      console.log("TITLE_ERROR: enter title of your project")
    }
    console.log('SAVE_PROJECT')
  }
  return (
    <button onClick={() => handleClick()}>Save</button>
  )
}
