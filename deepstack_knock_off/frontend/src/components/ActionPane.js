import { useState, useEffect } from "react";
import axios from 'axios';
import Action from "./Action";

const ActionPane = ({ gameState, fetchGameState, onWinnerDetermined, onRoundWinners, onAiDecision, aiDecision }) => {
  const [actions, setActions] = useState([]);
  const [selectedAction, setSelectedAction] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Determine the current player type
  const currentPlayerIsAI = gameState?.current_player?.name?.includes("AI");
  const currentPlayerIsHuman = gameState?.current_player?.name?.includes("Human");

  // Handle fetching AI decision or assigning legal actions based on the player type
  useEffect(() => {
    const handleGameUpdate = async () => {
      if (isProcessing) return;

      setIsProcessing(true);

      try {
        if (currentPlayerIsAI) {
          await fetchAIDecision();
        } else if (currentPlayerIsHuman && gameState.current_player.legal_actions == null) {
          await assignLegalActions();
        }
      } catch (error) {
        console.error('Error processing game update:', error);
      } finally {
        setIsProcessing(false);
      }
    };

    handleGameUpdate();
  }, [gameState, currentPlayerIsAI, currentPlayerIsHuman]);

  const fetchAIDecision = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/ai-decision");
      onAiDecision(response.data);
    } catch (error) {
      console.error("Failed to fetch AI decision:", error);
    }
  };

  const assignLegalActions = async () => {
    const playerName = gameState.current_player.name;
    console.log("Requesting actions assigned to player: ", playerName);
    const payload = { name: playerName };

    try {
      const response = await axios.post("http://127.0.0.1:5000/legal-actions", payload);
      if (response.status === 200) {
        console.log(response.data.message);
        fetchGameState();
      } else {
        console.error('Error assigning legal actions:', response.data.error);
      }
    } catch (error) {
      console.error('Failed to assign legal actions:', error);
    }
  };

  useEffect(() => {
    if (gameState?.current_player?.legal_actions) {
      setActions(gameState.current_player.legal_actions);
    }
  }, [gameState]);

  useEffect(() => {
    if (!aiDecision) return;

    const timer = setTimeout(() => {
      applyAction(aiDecision);
      onAiDecision(null);
    }, 4000);

    return () => clearTimeout(timer);
  }, [aiDecision, onAiDecision]);

  const applyAction = async (action) => {
    console.log("Applying action:", action);

    const payload = {
      name: action.action_name,
      player: action.player_name,
    };

    try {
      const response = await axios.post("http://127.0.0.1:5000/apply-action", payload);
      console.log("Action applied successfully:", response.data);

      if (response.data.winner) {
        onWinnerDetermined(response.data.winner);
        setActions([]);
      } else if (response.data.round_winners) {
        onRoundWinners(response.data.round_winners);
        setActions([]);
      } else {
        fetchGameState();
      }
    } catch (error) {
      console.error("Failed to apply action:", error);
    }
    setSelectedAction(null); // Reset selection after applying
  };

  const handleActionClick = (action) => {
    setSelectedAction(action);
  };

  const buttonClass = selectedAction ? "poker-button" : "poker-button disabled";

  return (
    <div className="action-pane">
      <div className="scroll-box">
        <ul>
          {!currentPlayerIsAI && actions.map((action, index) => (
            <Action
              key={index}
              action={action}
              onClick={() => handleActionClick(action)}
              isSelected={selectedAction === action}
            />
          ))}
        </ul>
      </div>
      <button className={buttonClass} onClick={() => applyAction(selectedAction)} disabled={!selectedAction || actions.length === 0}>Apply</button>
    </div>
  );
};

export default ActionPane;
