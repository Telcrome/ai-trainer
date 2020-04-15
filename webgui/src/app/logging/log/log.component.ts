import { Component, OnInit } from '@angular/core';
import { LogService } from '../log.service';
import { map } from 'rxjs/operators';
import { Log } from '../log';
import { CellComponent } from '../cell/cell.component';


@Component({
  selector: 'app-log',
  templateUrl: './log.component.html',
  styleUrls: ['./log.component.scss']
})
export class LogComponent implements OnInit {

  constructor(private logservice: LogService) { }

  logList: any[];

  ngOnInit(): void {
    // let mapper = map((logjsons: Array<any>) => logjsons.forEach(x => new Log(x)));
    this.logservice.getLogs().subscribe((x: any[]) => this.logList = Log.fromJson(x));
  }

}
