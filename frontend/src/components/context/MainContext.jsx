import { createContext, useEffect, useState } from "react";
import { useNodesState, useEdgesState } from 'reactflow';
import CustomNode from "../CustomNode/CustomNode";


export const MainContext = createContext()

const TestInitialNodes = [
    { id: '1', position: { x: 0, y: 0 }, data: {
      label: 'node 1',
      param1: 'value1',
      param2: 'value1',
      param3: 'value1',
      param4: 'value1',
      param5: 'value1',
      param6: 'value1',
    }},
    { id: '2', position: { x: 0, y: 100 }, data: { label: 'node 2', param1: 'value1'}},
  ];
const TestInitialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

const MainContextProvider = ({ children }) => {
  const [projects, setProjects] = useState([
    {
      name: 'test_project_1',
      edges: [
        {id: 'e1-2', source: '1', target: '2'}
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
         }}
      ],
      viewport: {x: 412, y: 247, zoom: 2},
    },
    {
      name: 'test_project_2',
      edges: [
        {id: 'e1-2', source: '1', target: '2'}
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
          position: {x: -180, y: 90},
          data: {
           label: 'node 3', param1: 'value1', param2: 'value1',
         }}
      ],
      viewport: {x: 412, y: 247, zoom: 2},
    }
  ])

  const [nodes, setNodes, onNodesChange] = useNodesState(TestInitialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(TestInitialEdges);

  const [savedNodes, setSavedNodes] = useState([
    {id: '1', data: { label: 'saved_node_1', param1: 'value1'}},
    {id: '2', data: { label: 'saved_node_2', param1: 'value1'}}
  ])
  const [nodeTypes, setNodeTypes] = useState({
    customNode1: CustomNode
  })

  const [currentNode, setCurrentNode] = useState('0')
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const rendered = false  //TODO: fix rerender
  useEffect(() => {
    const flow = JSON.parse(localStorage.getItem('load-flow'));

    if (flow) {
      setNodes(flow.nodes || []);
      setEdges(flow.edges || []);
    }
  }, [rendered])

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
      savedNodes,
      setSavedNodes,
      projects,
      setProjects,
    }}>
      {children}
    </MainContext.Provider>
  )
}

export default MainContextProvider