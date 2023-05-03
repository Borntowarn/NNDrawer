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
    onSave(flow)
    if (currentProject) {
      setProjects(projects.map((prj) =>
      prj.name === currentProject ? 
      {...prj, instance: {
        viewport: prj.instance.viewport,
        nodes: flow.nodes,
        edges: flow.edges,
      }} : prj ))
    }else {
      console.log("TITLE_ERROR: enter title of your project")
    }
  }


  return (
    <button onClick={() => handleClick()}>Save</button>
  )
}
