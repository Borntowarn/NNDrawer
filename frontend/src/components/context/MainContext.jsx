import { createContext, useState } from "react";
import { useNodesState, useEdgesState } from 'reactflow';

export const MainContext = createContext()

const nodeTypes = ['input', 'default', 'output', 'type1', 'type2', 'type3', 'type4']

const TestInitialNodes = [
    { id: '1', position: { x: 0, y: 0 }, data: { label: 'node 1', param1: 'value1', param2: 'value1' }},
    { id: '2', position: { x: 0, y: 100 }, data: { label: 'node 2', param1: 'value1'}},
    { id: '3', type: 'customNode', position: { x: 0, y: 200 }, data: 'Test'},
  ];
const TestInitialEdges = [{ id: 'e1-2', source: '1', target: '2' }];



const MainContextProvider = ({ children }) => {
    const [nodes, setNodes, onNodesChange] = useNodesState(TestInitialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(TestInitialEdges);

    const [currentNode, setCurrentNode] = useState('0')

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
            nodeTypes
        }}>
          {children}
        </MainContext.Provider>
    )
}

export default MainContextProvider