import spadesIcon from "../assets/icons8-spades-96.png"
import heartsIcon from "../assets/icons8-favorite-96.png"
import diamondsIcon from "../assets/icons8-diamonds-96.png";
import clubsIcon from "../assets/icons8-clubs-96.png"


const Card = ({ rank, suit }) => {
  let suitIcon = null;
  let rankColorClass = "";

  if (suit === "spades")
    suitIcon = spadesIcon;
  else if (suit === "hearts")
    suitIcon = heartsIcon;
  else if (suit === "diamonds")
    suitIcon = diamondsIcon;
  else if (suit === "clubs")
    suitIcon = clubsIcon;

  if (suit === "hearts" || suit === "diamonds")
    rankColorClass = "red-rank";
  else
    rankColorClass = "black-rank";

  return (
    <div className="card">
      <p className={`card-rank ${rankColorClass}`}>{rank}</p>
      <img className="card-suit" alt="card-suit" src={suitIcon} />
    </div>
  )
};

export default Card