import React, { useCallback, useRef, useMemo, useContext } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  Controls,
  Background,
  addEdge,
} from 'reactflow';
import { useOnSelectionChange } from 'reactflow';
import { MainContext } from '../../context/MainContext';
import { v4 as uuid } from 'uuid';

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
    nodeTypes,
  } = useContext(MainContext)

  useOnSelectionChange({
    onChange: ({ nodes, edges }) => console.log('changed selection', nodes, edges),
  });

  const onNodeClick = (event, node) => {
    console.log('SELECTED_NODE:', node)
    setCurrentNode(node)
  };
  
  const allowedTypes = useMemo(() => (nodeTypes), [nodeTypes]);

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
      const nodeData = JSON.parse(event.dataTransfer.getData('node'));

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode = {
        id:  nodeData.id ? nodeData.id + Date.now().toString() : Date.now().toString(),  // TODO: better to use slice of uuid
        position,
        type: nodeData.type,
        data: {...nodeData.params, label: nodeData.title}
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance]
  );

  return (
      <ReactFlowProvider>
        <ReactFlow
          ref={ reactFlowWrapper }
          nodes={nodes}
          edges={edges}
          nodeTypes={allowedTypes}
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