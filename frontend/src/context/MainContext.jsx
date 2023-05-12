import { createContext, useEffect, useState } from "react";
import { useNodesState, useEdgesState } from 'reactflow';
import CustomNode from "../components/CustomNode/CustomNode";


export const MainContext = createContext()

const MainContextProvider = ({ children }) => {
  const [projects, setProjects] = useState([
    {
      name: 'test_project_1',
      instance: {
        edges: [
          {id: 'e1-2', source: '1', target: '2'}
        ],
        nodes: [
          {width: 150,
           height: 40,
           id: '1',
           position: {x: -90, y: 0},
           data: {
            label: 'node 1', param1: 'value1', param2: 'value1'
          }},
          {width: 150,
            height: 40,
            id: '2',
            position: {x: -90, y: 90},
            data: {
             label: 'node 2', param1: 'value1', param2: 'value1'
           }}
        ],
        viewport: {x: 412, y: 247, zoom: 2}
      }
    },
    {
      name: 'test_project_2',
      instance: {
        edges: [
          {id: 'e1-2', source: '1', target: '3'}
        ],
        nodes: [
          {width: 150,
           height: 40,
           id: '1',
           position: {x: -90, y: 0},
           data: {
            label: 'node 1', param1: 'value1', param2: 'value1',
          }},
          {width: 150,
            height: 40,
            id: '2',
            position: {x: -90, y: 90},
            data: {
             label: 'node 2', param1: 'value1', param2: 'value1',
           }},
           {width: 150,
            height: 40,
            id: '3',
            position: {x: -280, y: 90},
            data: {
             label: 'node 3', param1: 'value1', param2: 'value1',
           }}
        ],
        viewport: {x: 412, y: 247, zoom: 2},
      }
    }
  ])

  const [authData, setAuth] = useState()
  const [authModalActive, setAuthModalActive] = useState(true)
  const [regModalActive, setRegModalActive] = useState(false)
  const [currentProject, setCurrentProject] = useState(null)
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [showedNodes, setShowedNodes] = useState([])
  const [customNodes, setCustomNodes] = useState([])


  const [nodeTypes, setNodeTypes] = useState({
    customNode1: CustomNode
  })

  const [currentNode, setCurrentNode] = useState('0')
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  // const rendered = false  //TODO: fix rerender
  useEffect(() => {
    const flow = localStorage.getItem('load-project');

    if (!!flow && projects.find(prj => prj.name === flow)) {
      const data = projects.find(elem => elem.name === flow)
      setNodes(data.instance.nodes || []);
      setEdges(data.instance.edges || []);
    }
  }, [])

  const updateInstance = (project) => {
    if (reactFlowInstance) {
      console.log('CHANGE INSTANCE')
      setCurrentProject(project.name)
      setNodes(project.instance.nodes)
      setEdges(project.instance.edges)
    }
  }

  const createNewProject = () => {
    setCurrentProject(null)
    setNodes([])
    setEdges([])
  }

  return (
    <MainContext.Provider value={{
      onNodesChange,
      onEdgesChange,
      nodes,
      setNodes,
      edges,
      setEdges,
      currentNode,
      setCurrentNode,
      nodeTypes,
      setNodeTypes,
      reactFlowInstance,
      setReactFlowInstance,
      showedNodes,
      setShowedNodes,
      projects,
      setProjects,
      currentProject,
      setCurrentProject,
      updateInstance,
      authModalActive,
      setAuthModalActive,
      regModalActive,
      setRegModalActive,
      authData,
      setAuth,
      createNewProject,
      customNodes,
      setCustomNodes
    }}>
      {children}
    </MainContext.Provider>
  )
}

export default MainContextProvider