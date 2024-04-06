import { useState, useEffect, useCallback } from "react";
import Action from "./Action";
import axios from 'axios';


const ActionPane = ({ gameState, fetchGameState, onWinnerDetermined, onRoundWinners, onAiDecision, aiDecision }) => {
  const [actions, setActions] = useState([]);
  const [selectedAction, setSelectedAction] = useState(null);
  const [isAssigningActions, setIsAssigningActions] = useState(false);

  const fetchAIDecision = useCallback(async () => {
    try {
      // Ensure it's AI's turn and it's the first call to this function
      if (gameState?.current_player?.name?.includes("AI")) {
        const response = await axios.get("http://127.0.0.1:5000/ai-decision")
        onAiDecision(response.data)
      }
    } catch (error) {
      console.error("Failed to fetch AI decision:", error);
    }
  }, [gameState, onAiDecision]);

  useEffect(() => {
    const assignLegalActions = async () => {
      // Early return if we're already assigning actions
      if (isAssigningActions || gameState?.current_player?.name.includes("AI")) return;

      try {
        setIsAssigningActions(true)
        if (gameState?.current_player?.name) {
          const playerName = gameState.current_player.name;
          console.log("Requesting actions assigned to player: ", playerName);
          const payload = { name: playerName };

          const response = await axios.post("http://127.0.0.1:5000/legal-actions", payload);
          if (response.status === 200) {
            console.log(response.data.message);
            fetchGameState(); // Fetch the latest gameState which should now include the assigned legal actions
          } else {
            console.error('Error assigning legal actions:', response.data.error);
          }
        }
      } catch (error) {
        console.error('Failed to assign legal actions:', error);
      } finally {
        setIsAssigningActions(false)
      }
    };
    // Fetch AI decision only when it's AI's turn
    if (gameState?.current_player?.name?.includes("AI")) {
      fetchAIDecision();
    } else if (gameState?.current_player?.name?.includes("Human") && gameState.current_player.legal_actions == null) {
      assignLegalActions();
    }
  }, [gameState, fetchAIDecision, fetchGameState, isAssigningActions]);

  useEffect(() => {
    // Assuming gameState.current_player.legal_actions is updated with new actions
    const legalActions = gameState?.current_player?.legal_actions;
    if (!legalActions) {
      return;
    }
    if (gameState.current_player && gameState.current_player.legal_actions) {
      setActions(gameState.current_player.legal_actions);
    }
  }, [gameState]);

  const applyAIAction = useCallback((action) => {
    console.log("Applying AI action:", action);
    const payload = {
      name: action.action_name,
      player: action.player_name
    };

    axios.post("http://127.0.0.1:5000/apply-action", payload)
      .then(response => {
        console.log("AI action applied successfully:", response.data);
        if (response.data.winner) {
          // A game winner has been determined
          onWinnerDetermined(response.data.winner)
          setActions([])
          setSelectedAction(null)
        } else if (response.data.round_winners) {
          onRoundWinners(response.data.round_winners)
          setActions([]);
          setSelectedAction(null)
        } else {
          fetchGameState();
        }
        onAiDecision(null); // Properly reset aiDecision here if not done elsewhere
      })
      .catch(error => {
        console.error("Failed to apply AI action:", error);
      });
      // Reset selection after applying
      setSelectedAction(null);

    }, [fetchGameState, onAiDecision, onWinnerDetermined, onRoundWinners]);

  useEffect(() => {
    if (aiDecision && aiDecision.action_name) {
      const timer = setTimeout(() => {
        console.log(aiDecision)
        applyAIAction(aiDecision);
        onAiDecision(null); // Reset aiDecision to null after applying
      }, 4000); // Wait 4 seconds
      return () => clearTimeout(timer); // Clean up the timeout if the component unmounts
    }
  }, [aiDecision, onAiDecision, applyAIAction]);

  const handleActionClick = (action) => {
    setSelectedAction(action);
  };

  const applySelectedAction = () => {
    console.log("Applying action:", selectedAction);

    const payload = {
      name: selectedAction.action_name,
      player: selectedAction.player_name
    };

    axios.post("http://127.0.0.1:5000/apply-action", payload)
      .then(response => {
        console.log("Action applied successfully:", response.data);
        if (response.data.winner) {
          // A game winner has been determined
          onWinnerDetermined(response.data.winner)
          setActions([]);
          setSelectedAction(null);
        } else if (response.data.round_winners) {
          onRoundWinners(response.data.round_winners)
          setActions([]);
          setSelectedAction(null);
        } else {
          fetchGameState()
        }
      })
      .catch(error => {
        console.error("Failed to apply action:", error);
      });
    // Reset selection after applying
    setSelectedAction(null);
  };

  const isAITurn = gameState?.current_player && gameState?.current_player.name.includes("AI");
  const buttonClass = selectedAction ? "poker-button" : "poker-button disabled";

  return (
    <div className="action-pane">
      <div className="scroll-box">
        <ul>
          {(!isAITurn && actions.length > 0) ? actions.map((action, index) => (
            <Action
              key={index}
              action={action}
              onClick={() => handleActionClick(action)}
              isSelected={selectedAction === action}
            />
          )) : null}
        </ul>
      </div>
      <button className={buttonClass} onClick={applySelectedAction} disabled={!selectedAction || actions.length === 0}>Apply</button>
    </div>
  );
};

export default ActionPane;
