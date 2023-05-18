import { createContext, useEffect, useState } from "react";
import { useNodesState, useEdgesState } from 'reactflow';
import CustomNode from "../components/CustomNode/CustomNode";
import axios from "axios";
import constants from "../constants/constants";

export const MainContext = createContext()

const MainContextProvider = ({ children }) => {
  const [projects, setProjects] = useState([])
  const [authData, setAuth] = useState()
  const [authModalActive, setAuthModalActive] = useState(true)
  const [regModalActive, setRegModalActive] = useState(false)
  const [currentProject, setCurrentProject] = useState(null)
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [showedNodes, setShowedNodes] = useState([])
  const [currentTitle, setCurrentTitle] = useState('')
  const [customNodes, setCustomNodes] = useState([
    {id: '0', position: { x: 0, y: 0 }, data: {label: 'start', Args: []}}
  ])

  const [nodeTypes, setNodeTypes] = useState({
    customNode1: CustomNode
  })

  const [currentNode, setCurrentNode] = useState('0')
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  // TODO: fix rerender
  useEffect(() => {
    checkToken()

    // const flow = localStorage.getItem('load-project');
    // if (!!flow && projects.find(prj => prj.name === flow)) {
    //   const data = projects.find(elem => elem.name === flow)
    //   setNodes(data.instance.nodes || []);
    //   setEdges(data.instance.edges || []);
    // }
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
    setNodes([{id: '0', position: { x: 0, y: 0 }, data: {label: 'start', Args: []}}])
    setEdges([])
  }

  const checkToken = async () => {
    const token = localStorage.getItem('user-token')
    if (token) {
      try {
        const response = axios.post(constants.urls.token, token)
        
        setAuth(response.user.id)
        const importProjects =  response.data.projects.map((elem) => {
          return JSON.parse(elem.data)
        })
        setProjects(importProjects)
        setAuthModalActive(false)
      } catch(err) {
        console.log("ERROR: ", err)
      }
    }
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
      setCustomNodes,
      currentTitle,
      setCurrentTitle,
    }}>
      {children}
    </MainContext.Provider>
  )
}

export default MainContextProvider