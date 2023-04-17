import React, { useCallback, useRef, useMemo, useState, useEffect, useContext } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  Controls,
  Background,
  addEdge,
} from 'reactflow';
import { useOnSelectionChange } from 'reactflow';
import { MainContext } from '../context/MainContext';

import CustomNode from '../CustomNode/CustomNode';

import 'reactflow/dist/style.css';
import '../DrawZone/DrawZone.css'


let id = 0;
const getId = () => `dndnode_${id++}`;

export default function DrawZone() {
  const {
    setCurrentNode,
    nodes, 
    edges, 
    setNodes, 
    onNodesChange, 
    setEdges, 
    onEdgesChange,
    rfInstance,
    setRfInstance
  } = useContext(MainContext)

  useOnSelectionChange({
    onChange: ({ nodes, edges }) => console.log('changed selection', nodes, edges),
  });

  const onNodeClick = (event, node) => {
    console.log('SELECTED_NODE:', node)
    setCurrentNode(node)
  };
  
  const nodeTypes = useMemo(() => ({ customNode: CustomNode }), []);
  
  const edgeUpdateSuccessful = useRef(true);
  const reactFlowWrapper = useRef(null);
  
  // const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const { reactFlowInstance, setReactFlowInstance } = useContext(MainContext);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);
  
  const onEdgeUpdateStart = useCallback(() => {
    edgeUpdateSuccessful.current = false;
  }, []);

  const onEdgeUpdate = useCallback((oldEdge, newConnection) => {
    edgeUpdateSuccessful.current = true;
    setEdges((els) => updateEdge(oldEdge, newConnection, els));
  }, []);

  const onEdgeUpdateEnd = useCallback((_, edge) => {
    if (!edgeUpdateSuccessful.current) {
      setEdges((eds) => eds.filter((e) => e.id !== edge.id));
    }
    edgeUpdateSuccessful.current = true;
  }, []);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      console.log("DROP_TYPE: ", type)

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });
      const newNode = {
        id: getId(),
        type,
        position,
        data: { label: `${type} node` },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance]
  );
  const reactFlowStyle = {
    
  }

  return (
      <ReactFlowProvider>
        <ReactFlow
          style = {reactFlowStyle}
          ref={reactFlowWrapper}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          snapToGrid
          onEdgeUpdate={onEdgeUpdate}
          onEdgeUpdateStart={onEdgeUpdateStart}
          onEdgeUpdateEnd={onEdgeUpdateEnd}
          onConnect={onConnect}
          onInit={setReactFlowInstance}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeClick={onNodeClick}
          fitView
          attributionPosition="top-right">
          <Controls />  
          <Background variant="dots" gap={12} size={0.5} />
        </ReactFlow>
      </ReactFlowProvider>  
  );
}