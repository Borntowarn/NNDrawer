import { useCallback, useContext } from 'react';
import { MainContext } from '../../context/MainContext';


export default function SaveButton() {
  const { reactFlowInstance, currentProject, setProjects, projects } = useContext(MainContext);
  const flowKey = 'load-project';

  const onSave = useCallback(() => {
    if (reactFlowInstance) {
      localStorage.setItem(flowKey, currentProject);
      }
  }, [reactFlowInstance, currentProject])

  const handleClick = () => {
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

    } else {
      console.log("TITLE_ERROR: enter title of your project")
    }
  }


  return (
    <button onClick={() => handleClick()}>Save</button>
  )
}
